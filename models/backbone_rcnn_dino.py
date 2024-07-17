import bisect
import json

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

from util.misc import NestedTensor
from .groundingdino.models.GroundingDINO import build_groundingdino
from .groundingdino.util.misc import clean_state_dict
from .groundingdino.util.slconfig import SLConfig
from .groundingdino.util.utils import get_phrases_from_posmap
from .groundingdino.util.inference import annotate
import groundingdino.datasets.transforms as T
import cv2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MaskRCNNDinoBackbone(MaskRCNN):
  def __init__(self, input_resize=(240, 320), n_classes=8, backbone_str='resnet50-fpn', dino_backbone=None,
               train_backbone=False, return_interm_layers=True, dataset='lmo',
               anchor_sizes=((32,), (64,), (128,), (256,), (512,)), class_info=None):

    assert backbone_str == 'resnet50-fpn'
    backbone = resnet_fpn_backbone('resnet50', pretrained=False)

    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    super().__init__(backbone=backbone, num_classes=n_classes, rpn_anchor_generator=rpn_anchor_generator,
                     max_size=max(input_resize), min_size=min(input_resize))

    assert dino_backbone is not None
    self.dino_backbone = dino_backbone

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
        self.map[value] = int(key)

    # self.map = {
    #   "master chef can": 1,
    #   "cracker box": 2,
    #   "sugar box": 3,
    #   "tomato soup can": 4,
    #   "mustard bottle": 5,
    #   "tuna fish can": 6,
    #   "pudding box": 7,
    #   "gelatin box": 8,
    #   "potted meat can": 9,
    #   "banana": 10,
    #   "pitcher base": 11,
    #   "bleach cleanser": 12,
    #   "bowl": 13,
    #   "mug": 14,
    #   "power drill": 15,
    #   "wood block": 16,
    #   "scissors": 17,
    #   "large marker": 18,
    #   "large clamp": 19,
    #   "extra large clamp": 20,
    #   "foam brick": 21
    # }

    self.vectorizer = TfidfVectorizer()
    self.tfidf = self.vectorizer.fit_transform(self.map.keys())

    self.dino_caption = ". ".join(list(self.map.keys()))

  def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
      return result
    return result + "."

  def dinoPredict(self, images: torch.Tensor, caption_: str, box_threshold: float, text_threshold: float,
                  device: str = "cuda", remove_combined: bool = False) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:

    # "preprocess_caption()"
    caption = caption_.lower().strip()
    if not caption.endswith("."):
      caption = caption + "."

    model = self.dino_backbone.to(device)
    images = images.to(device)

    with torch.no_grad():
      outputs = model(images, captions=[caption for _ in range(len(images))])

    for idx, _ in enumerate(range(outputs["pred_boxes"].shape[0])):
      prediction_logits = outputs["pred_logits"][0].cpu().sigmoid()  # prediction_logits.shape = (nq, 256)
      prediction_boxes = outputs["pred_boxes"][0].cpu()  # prediction_boxes.shape = (nq, 4)

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
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer, left_idx, right_idx).replace('.', ''))
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
    raw_images = tensor_list.tensors.cpu().numpy()

    predictions = []
    for idx, (boxes, logits, phrases) in enumerate(self.dinoPredict(images=images, caption_=self.dino_caption, box_threshold=0.35, text_threshold=0.25)):
      image = raw_images[idx]
      image = np.moveaxis(image, 0, 2)  # transform (c, h, w) to (h, w, c) (put rgb channel at the end)

      h, w, _ = image.shape # h, w, c
      boxes = boxes * torch.Tensor([w, h, w, h])
      boxes = box_convert(boxes, "cxcywh", "xyxy")  # Convert bbox to xyxy format (later in PoET it will be converted back to cxcywh)

      ################################
      # BBox Visualization
      #
      # labels = [
      #   f"{phrase} {logit:.2f}"
      #   for phrase, logit
      #   in zip(phrases, logits)
      # ]
      # detections = sv.Detections(xyxy=boxes.numpy())
      # box_annotator = sv.BoxAnnotator()
      # annotated_frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      # annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
      # cv2.imshow('frame', annotated_frame)
      # cv2.waitKey(-1)
      # cv2.destroyAllWindows()

      if not len(boxes) == 0:
        p = []
        for box, logits, label in zip(boxes, logits, phrases):
          label_vec = self.vectorizer.transform([label])
          cos_sim = cosine_similarity(label_vec, self.tfidf).flatten()

          # TODO: Refactor threshold
          if all(v <= 0.1 for v in cos_sim):  # If not a single label matches 30% of pred label
            continue

          best_match_idx = np.argmax(cos_sim)
          best_match = list(self.map.keys())[best_match_idx]
          cls = self.map[best_match]

          pred = torch.hstack((box, logits, torch.tensor(cls))).to("cuda")
          p.append(pred)

        if len(p) != 0:  # Only stack valid found predictions
          p = torch.stack(p)
          predictions.append(p)

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
  backbone = MaskRCNNDinoBackbone(input_resize=(rcnn_cfg["input_resize"][0], rcnn_cfg["input_resize"][1]),
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
