import bisect
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator, concat_box_prediction_layers
from torchvision.ops import box_convert
from typing import Dict, List, Tuple
import yaml
from pathlib import Path
import supervision as sv
from PIL import ImageDraw, Image, ImageFont

from util import logger
from util.misc import NestedTensor
from .groundingdino.models.GroundingDINO import build_groundingdino
from .groundingdino.util.misc import clean_state_dict
from .groundingdino.util.slconfig import SLConfig
from .groundingdino.util.utils import get_phrases_from_posmap
from .groundingdino.util.inference import annotate
import models.groundingdino.datasets.transforms as T
import cv2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MaskRCNNDinoBackbone(MaskRCNN):
  def __init__(self, args, input_resize=(240, 320), n_classes=8, backbone_str='resnet50-fpn', dino_backbone=None,
               train_backbone=False, return_interm_layers=True, dataset='lmo',
               anchor_sizes=((32,), (64,), (128,), (256,), (512,)), class_info=None):

    self.args = args
    self.device = args.device if args.device else "cuda"

    assert backbone_str == 'resnet50-fpn'
    backbone = resnet_fpn_backbone('resnet50', pretrained=False)

    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    super().__init__(backbone=backbone, num_classes=n_classes, rpn_anchor_generator=rpn_anchor_generator,
                     max_size=max(input_resize), min_size=min(input_resize))

    assert dino_backbone is not None
    self.dino_backbone = dino_backbone

    self.class_mode = args.class_mode if args.class_mode else "specific"

    if return_interm_layers:
      # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
      self.return_layers = ['2', '3', 'pool']
      # Might be wrong
      self.strides = [8, 16, 32]
      self.num_channels = [256, 256, 256]
    else:
      self.return_layers = ['pool']
      self.strides = [256]
      self.num_channels = [256]

    # Freeze backbone if it should not be trained
    self.train_backbone = train_backbone
    if not train_backbone:
      for name, parameter in self.named_parameters():
        parameter.requires_grad_(False)

    # For the LMO set we need to map the object ids correctly.
    self.obj_id_map = None
    if dataset == 'lmo':
      self.obj_id_map = {1: 1, 5: 2, 6: 3, 8: 4, 9: 5, 10: 6, 11: 7, 12: 8}

    if args.class_info[0] == "/":
        args.class_info = args.class_info[1:]

    class_info = os.path.join(args.dataset_path, args.class_info)
    assert class_info is not None

    with open(class_info, 'r') as f:
      self.class_info = json.load(f)

    self.map = {}
    if dataset == "ycbv":
      for key, value in self.class_info.items():
        new_key = value[4:].replace('_', ' ')
        self.map[new_key] = int(key)
    else:
      for key, value in self.class_info.items():
        new_key = value.replace('_', ' ')
        self.map[new_key] = int(key)

    self.dino_caption = None
    self.token_spans = None

    if self.class_mode == "agnostic":
        if not args.dino_caption:
            logger.warn(f"Class mode is '{self.class_mode}', using default dino caption 'object in the middle.'!")
            self.dino_caption = "object in the middle."
            # self.token_spans = [[[0,6]]]
        else:
            self.dino_caption = args.dino_caption
    else:
        if args.dino_caption:
            logger.warn(f"Class mode is '{self.class_mode}', ignoring provided dino caption '{args.dino_caption}'!")

        d = []
        for key, val in self.class_info.items():
            v = val
            if dataset == "ycbv":
                v = v[4:]

            v = v.replace('_', ' ')
            d.append({
                "id": int(key),
                "name": v,
            })

        tokens, self.dino_caption = self.build_id2posspan_and_caption(d)

        self.token_spans = []
        for id_spans in tokens.values():
            self.token_spans.append(id_spans)

        # self.dino_caption = " . ".join(list(self.map.keys()))
        # self.dino_caption = self.dino_caption.replace("_", " ")
    logger.info(f"Dino caption: {self.dino_caption}")


  def create_positive_map_from_span(self, tokenized, token_span, max_text_len=256):
      """construct a map such that positive_map[i,j] = True iff box i is associated to token j
      Input:
          - tokenized:
              - input_ids: Tensor[1, ntokens]
              - attention_mask: Tensor[1, ntokens]
          - token_span: list with length num_boxes.
              - each item: [start_idx, end_idx]
      """
      positive_map = torch.zeros((len(token_span), max_text_len), dtype=torch.float)
      for j, tok_list in enumerate(token_span):
          for (beg, end) in tok_list:
              beg_pos = tokenized.char_to_token(beg)
              end_pos = tokenized.char_to_token(end - 1)
              if beg_pos is None:
                  try:
                      beg_pos = tokenized.char_to_token(beg + 1)
                      if beg_pos is None:
                          beg_pos = tokenized.char_to_token(beg + 2)
                  except:
                      beg_pos = None
              if end_pos is None:
                  try:
                      end_pos = tokenized.char_to_token(end - 2)
                      if end_pos is None:
                          end_pos = tokenized.char_to_token(end - 3)
                  except:
                      end_pos = None
              if beg_pos is None or end_pos is None:
                  continue

              assert beg_pos is not None and end_pos is not None
              if os.environ.get("SHILONG_DEBUG_ONLY_ONE_POS", None) == "TRUE":
                  positive_map[j, beg_pos] = 1
                  break
              else:
                  positive_map[j, beg_pos: end_pos + 1].fill_(1)

      return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)

  def build_captions_and_token_span(self, cat_list, force_lowercase):
      """
      Return:
          captions: str
          cat2tokenspan: dict
              {
                  'dog': [[0, 2]],
                  ...
              }
      """

      cat2tokenspan = {}
      captions = ""
      for catname in cat_list:
          class_name = catname
          if force_lowercase:
              class_name = class_name.lower()
          if "/" in class_name:
              class_name_list: List = class_name.strip().split("/")
              class_name_list.append(class_name)
              class_name: str = random.choice(class_name_list)

          tokens_positive_i = []
          subnamelist = [i.strip() for i in class_name.strip().split(" ")]
          for subname in subnamelist:
              if len(subname) == 0:
                  continue
              if len(captions) > 0:
                  captions = captions + " "
              strat_idx = len(captions)
              end_idx = strat_idx + len(subname)
              tokens_positive_i.append([strat_idx, end_idx])
              captions = captions + subname

          if len(tokens_positive_i) > 0:
              captions = captions + " ."
              cat2tokenspan[class_name] = tokens_positive_i

      return captions, cat2tokenspan

  def build_id2posspan_and_caption(self, category_dict: dict):
      """Build id2pos_span and caption from category_dict

      Args:
          category_dict (dict): category_dict
      """
      cat_list = [item["name"].lower() for item in category_dict]
      id2catname = {item["id"]: item["name"].lower() for item in category_dict}
      caption, cat2posspan = self.build_captions_and_token_span(cat_list, force_lowercase=True)
      id2posspan = {catid: cat2posspan[catname] for catid, catname in id2catname.items()}
      return id2posspan, caption

  def plot_boxes_to_image(self, image_pil, tgt):
      H, W = tgt["size"]
      boxes = tgt["boxes"]
      labels = tgt["labels"]
      assert len(boxes) == len(labels), "boxes and labels must have same length"

      draw = ImageDraw.Draw(image_pil)
      mask = Image.new("L", image_pil.size, 0)
      mask_draw = ImageDraw.Draw(mask)

      # draw boxes and masks
      for box, label in zip(boxes, labels):
          # from 0..1 to 0..W, 0..H
          box = box * torch.Tensor([W, H, W, H])
          # from xywh to xyxy
          box[:2] -= box[2:] / 2
          box[2:] += box[:2]
          # random color
          color = tuple(np.random.randint(0, 255, size=3).tolist())
          # draw
          x0, y0, x1, y1 = box
          x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

          draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
          # draw.text((x0, y0), str(label), fill=color)

          font = ImageFont.load_default()
          if hasattr(font, "getbbox"):
              bbox = draw.textbbox((x0, y0), str(label), font)
          else:
              w, h = draw.textsize(str(label), font)
              bbox = (x0, y0, w + x0, y0 + h)
          # bbox = draw.textbbox((x0, y0), str(label))
          draw.rectangle(bbox, fill=color)
          draw.text((x0, y0), str(label), fill="white")

          mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

      return image_pil, mask

  def dinoPredict(self, images: torch.Tensor, caption_: str, box_threshold: float, text_threshold: float,
                  remove_combined: bool = False) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:

    # "preprocess_caption()"
    caption = caption_.lower().strip()
    if not caption.endswith("."):
      caption = caption + "."

    model = self.dino_backbone.to(self.device)
    images = images.to(self.device)

    with torch.no_grad():
      outputs = model(images, captions=[caption for _ in range(len(images))])

    for idx, _ in enumerate(range(outputs["pred_boxes"].shape[0])):
      prediction_logits = outputs["pred_logits"][idx].sigmoid()  # prediction_logits.shape = (nq, 256)
      prediction_boxes = outputs["pred_boxes"][idx]  # prediction_boxes.shape = (nq, 4)


      # If token_spans given, then we are in "category"/"phrase" mode
      if self.token_spans:
          positive_maps = self.create_positive_map_from_span(
              model.tokenizer(caption_),
              token_span=self.token_spans
          ).to(self.device)  # n_phrase, 256

          logits_for_phrases = positive_maps @ prediction_logits.T  # n_phrase, nq
          all_logits = []
          all_phrases = []
          all_boxes = []
          for (token_span, logit_phr) in zip(self.token_spans, logits_for_phrases):
              # get phrase
              phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
              # get mask
              filt_mask = logit_phr > box_threshold
              if torch.all(filt_mask == False).cpu().item(): continue

              # filt box
              all_boxes.append(prediction_boxes[filt_mask])
              # filt logits
              all_logits.append(logit_phr[filt_mask])
              logit_phr_num = logit_phr[filt_mask]
              all_phrases.extend([phrase for _ in logit_phr_num])
              # all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])

          if len(all_boxes) == 0: # If not a single object detected
              yield [], [], []
          else:
              boxes_filt = torch.cat(all_boxes, dim=0).to(self.device)
              logits_filt = torch.cat(all_logits, dim=0).to(self.device)
              pred_phrases = all_phrases

              yield boxes_filt, logits_filt, pred_phrases
      else:
          # If token_spans not given, we are in the mode of describing the object's appearance
          mask = prediction_logits.max(dim=1)[0] > box_threshold
          logits = prediction_logits[mask]  # logits.shape = (n, 256)
          boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

          tokenizer = model.tokenizer
          tokenized = tokenizer(caption)

          if remove_combined:
            sep_idx = [i for i in range(len(tokenized['input_ids'])) if tokenized['input_ids'][i] in [101, 102, 1012]]

            phrases = []
            for logit in logits:
              max_idx = logit.argmax()
              insert_idx = bisect.bisect_left(sep_idx, max_idx)
              right_idx = sep_idx[insert_idx]
              left_idx = sep_idx[insert_idx - 1]
              phrases.append(
                get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer, left_idx, right_idx).replace('.',
                                                                                                                   ''))
          else:
            phrases = [
              get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
              for logit
              in logits
            ]

          yield boxes, logits.max(dim=1)[0], phrases

  def normalizeImages(self, images):
    # Inference image preprocessing from GroundingDINO
    transform = T.Compose(
      [
        # T.RandomResize([800], max_size=1333),
        # T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
      ]
    )
    transformed, _ = transform(images, None)
    return transformed

  def forward(self, tensor_list: NestedTensor):
    # image_sizes = [img.shape[-2:] for img in tensor_list.tensors]
    # xs = self.backbone.body(tensor_list.tensors)
    features = self.backbone(tensor_list.tensors)
    # predictions, _ = self.rpn(tensor_list.tensors, features)

    # # Generate proposals using the RPN
    # feature_maps = list(features.values())
    # objectness, pred_bbox_deltas = self.rpn.head(feature_maps)
    # grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
    # image_size = tensor_list.tensors.shape[-2:]
    # dtype, device = feature_maps[0].dtype, feature_maps[0].device
    # strides = [[torch.tensor(image_size[0] // g[0], dtype=torch.int64, device=device),
    #             torch.tensor(image_size[1] // g[1], dtype=torch.int64, device=device)] for g in grid_sizes]
    # self.rpn.anchor_generator.set_cell_anchors(dtype, device)
    # anchors_over_all_feature_maps = self.rpn.anchor_generator.grid_anchors(grid_sizes, strides)
    # anchors: List[List[torch.Tensor]] = []
    # for _ in range(len(tensor_list.tensors)):
    #     anchors_in_image = [anchors_per_feature_map for anchors_per_feature_map in anchors_over_all_feature_maps]
    #     anchors.append(anchors_in_image)
    # anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
    #
    # num_images = len(anchors)
    # num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
    # num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
    # objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
    # # apply pred_bbox_deltas to anchors to obtain the decoded proposals
    # # note that we detach the deltas because Faster R-CNN do not backprop through
    # # the proposals
    # proposals = self.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
    # proposals = proposals.view(num_images, -1, 4)
    # boxes, scores = self.rpn.filter_proposals(proposals, objectness, image_sizes, num_anchors_per_level)
    # detections, _ = self.roi_heads(features, boxes, image_sizes)
    #
    # # Translate the detections to predictions: [bbox, score, label]
    # # TODO: optimize code
    # predictions = []
    # for img_detections in detections:
    #     img_predictions = []
    #     for c, cls in enumerate(img_detections['labels']):
    #         box = img_detections["boxes"][c]
    #         box = torch.hstack((box, img_detections["scores"][c]))
    #         if self.obj_id_map is not None:
    #             if cls.item() in self.obj_id_map.keys():
    #                 new_cls = self.obj_id_map[cls.item()]
    #                 box = torch.hstack((box, torch.tensor(new_cls, dtype=torch.float32, device=device)))
    #             else:
    #                 # Processing object that has a label that is not in the object_id_map --> skip
    #                 continue
    #         else:
    #             box = torch.hstack((box, cls))
    #         img_predictions.append(box)
    #     if len(img_predictions) == 0:
    #         # Either no objects present or no detected --> Append None
    #         img_predictions = None
    #     else:
    #         img_predictions = torch.stack(img_predictions)
    #     predictions.append(img_predictions)

    # TODO: Check what todo in case of no detected object
    # TODO: Support multiple images

    # [bbox, score, label]
    images = self.normalizeImages(tensor_list.tensors)
    raw_images = tensor_list.tensors

    predictions = []
    for idx, (boxes_, logits, phrases) in enumerate(self.dinoPredict(images=images, caption_=self.dino_caption, box_threshold=self.args.dino_box_threshold, text_threshold=self.args.dino_txt_threshold)):
      image = raw_images[idx]
      # image = np.moveaxis(image, 0, 2)  # transform (c, h, w) to (h, w, c) (put rgb channel at the end)

      if len(boxes_) == 0:
          # If not a single object predicted in image, add dummy
          predictions.append(None)
          continue

      _, h, w = image.shape # h, w, c
      boxes = boxes_ * torch.Tensor([w, h, w, h]).to(self.device)
      boxes = box_convert(boxes, "cxcywh", "xyxy")  # Convert bbox to xyxy format (later in PoET it will be converted back to cxcywh)

      ################################
      # BBox Visualization
      #
      if self.args.dino_bbox_viz:
      # if True:
          pred_dict = {
              "boxes": boxes_.cpu(),
              "size": [h, w],  # H,W
              "labels": phrases,
          }
          img = (np.moveaxis(image.cpu().numpy(), 0, -1) * 255).astype('uint8')
          img = Image.fromarray(img)
          image_with_box, _ = self.plot_boxes_to_image(img, pred_dict)
          image_with_box.show()
          image_with_box.close()


          # labels = [
          #   f"{phrase} {logit:.2f}"
          #   for phrase, logit
          #   in zip(phrases, logits)
          # ]
          # detections = sv.Detections(xyxy=boxes.cpu().numpy(), class_id=np.zeros((len(labels),), np.int32))
          # box_annotator = sv.BoundingBoxAnnotator()
          # img = np.moveaxis(image.cpu().numpy(), 0, -1)
          # annotated_frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
          # annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
          # plt.imshow(annotated_frame)
          # plt.show()
          # cv2.imshow('frame', annotated_frame)
          # cv2.waitKey(-1)
          # cv2.destroyAllWindows()

      if not len(boxes) == 0:
        p = []
        for box, logits, label in zip(boxes, logits, phrases):
          pred = None
          if self.class_mode == "specific":
            cls = self.map[label]
            pred = torch.hstack((box, logits, torch.tensor(cls).to("cuda")))
          else:
            pred = torch.hstack((box, logits, torch.tensor(-1).to("cuda")))
          p.append(pred)

        if len(p) != 0:
          p = torch.stack(p)
          predictions.append(p)
        else:
          predictions.append(None)
      else:
        predictions.append(None)

    # Prepare the feature map
    out: Dict[str, NestedTensor] = {}
    for name in self.return_layers:
      x = features[name]
      m = tensor_list.mask
      assert m is not None
      mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
      out[name] = NestedTensor(x, mask)
    return predictions, out


def build_rcnn_dino(args):
  args_dino = SLConfig.fromfile("/home/sebastian/repos/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
  args_dino.device = "cuda"
  dino = build_groundingdino(args_dino)
  checkpoint = torch.load("/home/sebastian/repos/GroundingDINO/weights/groundingdino_swint_ogc.pth",
                          map_location="cpu")
  dino.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
  dino.eval()

  train_backbone = args.lr_backbone > 0
  return_interm_layers = (args.num_feature_levels > 1)
  rcnn_cfg = yaml.load(Path(args.backbone_cfg).read_text(), Loader=yaml.FullLoader)
  n_classes = len(rcnn_cfg["label_to_category_id"])
  backbone = MaskRCNNDinoBackbone(args=args, input_resize=(rcnn_cfg["input_resize"][0], rcnn_cfg["input_resize"][1]),
                                  dataset=args.dataset,
                                  n_classes=n_classes,
                                  backbone_str=rcnn_cfg["backbone_str"],
                                  dino_backbone=dino,
                                  class_info=args.dataset_path + args.class_info)

  if args.backbone_weights is not None:
    ckpt = torch.load(args.backbone_weights)
    if args.backbone == "maskrcnn":
      ckpt = ckpt['state_dict']
      backbone.load_state_dict(ckpt)
    elif args.backbone == "fasterrcnn":
      ckpt = ckpt['model']
      missing_keys, unexpected_keys = backbone.load_state_dict(ckpt, strict=False)
      if len(missing_keys) > 0:
        print("Loading Faster R-CNN weights")
        print('Missing Keys: {}'.format(missing_keys))
        print('PoET does not rely on the mask head of Mask R-CNN')
  return backbone
