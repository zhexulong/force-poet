# ------------------------------------------------------------------------
# PoET: Pose Estimation Transformer for Single-View, Multi-Object 6D Pose Estimation
# Copyright (c) 2022 Thomas Jantos (thomas.jantos@aau.at), University of Klagenfurt - Control of Networked Systems (CNS). All Rights Reserved.
# Licensed under the BSD-2-Clause-License with no commercial use [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE_DEFORMABLE_DETR in the LICENSES folder for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list)
from .backbone import build_backbone
from .matcher import build_matcher
from .deformable_transformer import build_deforamble_transformer
from .position_encoding import BoundingBoxEmbeddingSine
from .graph_transformer import GraphTransformer
from .force_matrix_head import ForceMatrixHead
import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class PoET(nn.Module):
    """
    Pose Estimation Transformer module that performs 6D, multi-object relative pose estimation.
    """
    def __init__(self, backbone, transformer, num_queries, num_feature_levels, n_classes, bbox_mode='gt',
                 ref_points_mode='bbox', query_embedding_mode='bbox', rotation_mode='6d', class_mode='agnostic',
                 aux_loss=True, backbone_type="yolo", use_graph_transformer=True, graph_hidden_dim=None,
                 graph_num_layers=4, graph_num_heads=4, use_force_prediction=True, force_scale_factor=100.0):
        """
        Initalizing the model.
        Parameters:
            backbone: torch module of the backbone to be used. Includes backbone and positional encoding.
            transformer: torch module of the transformer architecture
            num_queries: number of queries that the transformer receives. Is equal to the number of expected objects
            in the image
            num_feature_levels: number of feature levels that serve as input to the transformer.
            n_classes: number of classes present in the dataset.
            bbox_mode: mode that determines which and how bounding box information is fed into the transformer.
            ref_points_mode: mode that defines how the transformer determines the reference points.
            query_embedding_mode: mode that defines how the query embeddings are determined.
            rotation_mode: determines the rotation representation
            class_mode: determines whether PoET is trained class specific or agnostic
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            backbone_type: object detector backbone type
            use_graph_transformer: whether to use graph transformer for force prediction
            graph_hidden_dim: hidden dimension for graph transformer
            graph_num_layers: number of layers in graph transformer
            graph_num_heads: number of attention heads in graph transformer
            use_force_prediction: whether to predict forces
        """
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        self.backbone = backbone
        self.backbone_type = backbone_type
        self.aux_loss = aux_loss
        self.n_queries = num_queries
        self.n_classes = n_classes + 1  # +1 for dummy/background class
        self.bbox_mode = bbox_mode
        self.ref_points_mode = ref_points_mode
        self.query_embedding_mode = query_embedding_mode
        self.rotation_mode = rotation_mode
        self.class_mode = class_mode
        
        # Graph Transformer and Force Prediction settings
        self.use_graph_transformer = use_graph_transformer
        self.use_force_prediction = use_force_prediction
        self.graph_hidden_dim = graph_hidden_dim if graph_hidden_dim is not None else hidden_dim
        self.force_scale_factor = force_scale_factor

        # Determine Translation and Rotation head output dimension
        self.t_dim = 3
        if self.rotation_mode == '6d':
            self.rot_dim = 6
        elif self.rotation_mode in ['quat', 'silho_quat']:
            self.rot_dim = 4
        else:
            raise NotImplementedError('Rotational representation is not supported.')

        # Mass embedding for incorporating object mass into features
        if self.use_force_prediction:
            self.mass_embed = nn.Linear(1, hidden_dim // 4)  # Mass embedding
        
        # Force Matrix Head for pairwise force prediction
        if self.use_force_prediction:
            self.force_matrix_head = ForceMatrixHead(
                in_dim=hidden_dim,
                hidden_dim=self.graph_hidden_dim,
                num_heads=graph_num_heads,
                num_layers=graph_num_layers,
                edge_features='concat',
                dropout=0.1,
                mass_embed_dim=hidden_dim // 4
            )

        # Translation & Rotation Estimation Head
        if self.class_mode == 'agnostic':
            self.translation_head = MLP(hidden_dim, hidden_dim, self.t_dim, 3)
            self.rotation_head = MLP(hidden_dim, hidden_dim, self.rot_dim, 3)
        elif self.class_mode == 'specific':
            self.translation_head = MLP(hidden_dim, hidden_dim, self.t_dim * self.n_classes, 3)
            self.rotation_head = MLP(hidden_dim, hidden_dim, self.rot_dim * self.n_classes, 3)
        else:
            raise NotImplementedError('Class mode is not supported.')

        self.num_feature_levels = num_feature_levels
        if num_feature_levels > 1:
            # Use multi-scale features as input to the transformer
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            # If multi-scale then every intermediate backbone feature map is returned
            for n in range(num_backbone_outs):
                in_channels = backbone.num_channels[n]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            # If more feature levels are required than backbone feature maps are available then the last feature map is
            # passed through an additional 3x3 Conv layer to create a new feature map.
            # This new feature map is then used as the baseline for the next feature map to calculate
            # For details refer to the Deformable DETR paper's appendix.
            for n in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            # We only want to use the backbones last feature embedding map.
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )
            ])

        # Initialize the projection layers
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # Pose is predicted for each intermediate decoder layer for training with auxiliary losses
        # Only the prediction from the final layer will be used for the final pose estimation
        num_pred = transformer.decoder.num_layers
        self.translation_head = nn.ModuleList([copy.deepcopy(self.translation_head) for _ in range(num_pred)])
        self.rotation_head = nn.ModuleList([copy.deepcopy(self.rotation_head) for _ in range(num_pred)])
        
        # Force matrix prediction for each intermediate decoder layer (if enabled)
        if self.use_force_prediction:
            self.force_matrix_head = nn.ModuleList([copy.deepcopy(self.force_matrix_head) for _ in range(num_pred)])

        # Positional Embedding for bounding boxes to generate query embeddings
        if self.query_embedding_mode == 'bbox':
            self.bbox_embedding = BoundingBoxEmbeddingSine(num_pos_feats=hidden_dim / 8)
        elif self.query_embedding_mode == 'learned':
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
            # TODO: Optimize Code to not generate bounding box query embeddings, when query embed is in learning mode.
            self.bbox_embedding = BoundingBoxEmbeddingSine(num_pos_feats=hidden_dim / 8)
        else:
            raise NotImplementedError('This query embedding mode is not implemented.')

    def forward(self, samples: NestedTensor, targets=None):
        """
        Function expects a NestedTensor, which consists of:
            - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
            - samples.mask: a binary mask of shape [batch_size X H x W], containing 1 on padded pixels

        Functions expects a list of length batch_size, where each element is a dict with the following entries:
            - boxes: tensor of size [n_obj, 4], contains the bounding box (x_c, y_c, w, h) of each object in each image
            normalized to image size
            - labels: tensor of size [n_obj, ], contains the label of each object in the image
            - image_id: tensor of size [1],  contains the image id to which this annotation belongs to
            - relative_position; tensor of size [n_obj, 3], contains the relative translation for each object present
            in the image w.r.t the camera.
            - relative_rotation: tensor of size [n_obj, 3, 3], contains the relative rotation for each object present
            in the image w.r.t. the camera.


        It returns a dict with the following elements:
            - pred_translation: tensor of size [batch_size, n_queries, 3], predicted relative translation for each
            object query w.r.t. camera
            - pred_rotation: tensor of size [batch_size, n_queries, 3, 3], predicted relative rotation for each
            object query w.r.t. camera
            - pred_boxes: tensor of size [batch_size, n_queries, 4], predicted bounding boxes (x_c, y_c, w, h) for each
            object query normalized to the image size
            - pred_classes: tensor of size [batch_size, n_queries], predicted class for each
            object query
            - aux_outputs: Optional, only returned when auxiliary losses are activated. It is a list of dictionaries
            containing the output values for each decoder layer.

        It returns a list "n_boxes_per_sample" of length [batch_size, 1], which contains the number of
        """

        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        # Store the image size in HxW
        image_sizes = [[sample.shape[-2], sample.shape[-1]] for sample in samples.tensors]
        features, pos, pred_objects = self.backbone(samples)

        assert samples.tensors.shape[0] == len(pred_objects), "Number of predictions must match number of images!"

        # TODO: Refactor case of no predictions
        if len(pred_objects) == 0:
            return None, None

        # Extract the bounding boxes for each batch element
        pred_boxes = []
        pred_classes = []
        query_embeds = []
        n_boxes_per_sample = []

        # Depending on the bbox mode, we either use ground truth bounding boxes or backbone predicted bounding boxes for
        # transformer query input embedding calculation.
        if self.bbox_mode in ['gt', 'jitter'] and targets is not None:
            for t, target in enumerate(targets):
                # GT from COCO loaded as x1,y1,x2,y2, but by data loader transformed to cx, cy, w, h and normalized
                if self.bbox_mode == 'gt':
                    t_boxes = target["boxes"]
                elif self.bbox_mode == 'jitter':
                    t_boxes = target["jitter_boxes"]
                n_boxes = len(t_boxes)
                n_boxes_per_sample.append(n_boxes)

                # Add classes
                t_classes = target["labels"]

                # For the current number of boxes determine the query embedding
                query_embed = self.bbox_embedding(t_boxes)
                # As the embedding will serve as the query and key for attention, duplicate it to be later splitted
                query_embed = query_embed.repeat(1, 2)

                # We always predict a fixed number of object poses per image set to the maximum number of objects
                # present in a single image throughout the whole dataset. Check whether this upper limit is reached,
                # otherwise fill up with dummy embeddings that are defined as cx,cy,w,h = [-1, -1, -1, -1]
                # Dummy boxes will later be filtered out by the matcher and not used for cost calculation
                if n_boxes < self.n_queries:
                    dummy_boxes = torch.tensor([[-1, -1, -1, -1] for i in range(self.n_queries-n_boxes)],
                                               dtype=torch.float32, device=t_boxes.device)

                    dummy_embed = torch.tensor([[-10] for i in range(self.n_queries-n_boxes)],
                                               dtype=torch.float32, device=t_boxes.device)
                    dummy_embed = dummy_embed.repeat(1, self.hidden_dim*2)
                    t_boxes = torch.vstack((t_boxes, dummy_boxes))
                    query_embed = torch.cat([query_embed, dummy_embed], dim=0)
                    dummy_classes = torch.tensor([-1 for i in range(self.n_queries-n_boxes)],
                                               dtype=torch.int, device=t_boxes.device)
                    t_classes = torch.cat((t_classes, dummy_classes))
                pred_boxes.append(t_boxes)
                query_embeds.append(query_embed)
                pred_classes.append(t_classes)
        elif self.bbox_mode == 'backbone':
            # Prepare the output predicted by the backbone
            # Iterate over batch and prepare each image in batch
            for bs, predictions in enumerate(pred_objects):
                if predictions is None:
                    # Case: Backbone has not predicted anything for image
                    # Add only dummy boxes, but mark that nothing has been predicted
                    n_boxes = 0
                    n_boxes_per_sample.append(n_boxes)
                    backbone_boxes = torch.tensor([[-1, -1, -1, -1] for i in range(self.n_queries - n_boxes)],
                                                  dtype=torch.float32, device=features[0].decompose()[0].device)
                    query_embed = torch.tensor([[-10] for i in range(self.n_queries - n_boxes)],
                                               dtype=torch.float32, device=features[0].decompose()[0].device)
                    query_embed = query_embed.repeat(1, self.hidden_dim * 2)
                    backbone_classes = torch.tensor([-1 for i in range(self.n_queries - n_boxes)], dtype=torch.int64,
                                                    device=features[0].decompose()[0].device)
                else:
                    # Case: Backbone predicted something
                    backbone_boxes = predictions[:, :4]
                    backbone_boxes = box_ops.box_xyxy_to_cxcywh(backbone_boxes)
                    # TODO: Adapt to different image sizes as we assume constant image size across the batch
                    backbone_boxes = box_ops.box_normalize_cxcywh(backbone_boxes, image_sizes[0])
                    n_boxes = len(backbone_boxes)

                    # Predicted classes by backbone // class 0 is "background"
                    # Scores predicted by the backbone are needed for top-k selection
                    backbone_scores = predictions[:, 4]
                    backbone_classes = predictions[:, 5]
                    backbone_classes = backbone_classes.type(torch.int64)

                    # For the current number of boxes determine the query embedding
                    query_embed = self.bbox_embedding(backbone_boxes)
                    # As the embedding will serve as the query and key for attention, duplicate it to be later splitted
                    query_embed = query_embed.repeat(1, 2)

                    if n_boxes < self.n_queries:
                        # Fill up with dummy boxes to match the query size and add dummy embeddings
                        dummy_boxes = torch.tensor([[-1, -1, -1, -1] for i in range(self.n_queries - n_boxes)],
                                                   dtype=torch.float32, device=backbone_boxes.device)
                        dummy_embed = torch.tensor([[-10] for i in range(self.n_queries - n_boxes)],
                                                   dtype=torch.float32, device=backbone_boxes.device)
                        dummy_embed = dummy_embed.repeat(1, self.hidden_dim * 2)
                        backbone_boxes = torch.cat([backbone_boxes, dummy_boxes], dim=0)
                        query_embed = torch.cat([query_embed, dummy_embed], dim=0)
                        dummy_classes = torch.tensor([-1 for i in range(self.n_queries - n_boxes)],
                                                     dtype=torch.int64, device=backbone_boxes.device)
                        backbone_classes = torch.cat([backbone_classes, dummy_classes], dim=0)
                    elif n_boxes > self.n_queries:
                        # Number of boxes will be limited to the number of queries
                        n_boxes = self.n_queries
                        # Case: backbone predicts more output objects than queries available --> take top n_queries
                        # Sort scores to get the post top performing ones
                        backbone_scores, indices = torch.sort(backbone_scores, dim=0, descending=True)
                        backbone_classes = backbone_classes[indices]
                        backbone_boxes = backbone_boxes[indices, :]
                        query_embed = query_embed[indices, :]

                        # Take the top n predictions
                        backbone_scores = backbone_scores[:self.n_queries]
                        backbone_classes = backbone_classes[:self.n_queries]
                        backbone_boxes = backbone_boxes[:self.n_queries]
                        query_embed = query_embed[:self.n_queries]
                    n_boxes_per_sample.append(n_boxes)
                pred_boxes.append(backbone_boxes)
                pred_classes.append(backbone_classes)
                query_embeds.append(query_embed)
        else:
            raise NotImplementedError("PoET Bounding Box Mode not implemented!")

        query_embeds = torch.stack(query_embeds)  # "object queries" => generated embeddings from the bbox prediction
        pred_boxes = torch.stack(pred_boxes)
        pred_classes = torch.stack(pred_classes)

        # Construct multi-scale feature maps from the original feature maps of the backbone,
        # for subsequent processing by the Deformable DETR transformer.
        srcs = []
        masks = []
        for lvl, feat in enumerate(features):
            # Iterate over each feature map of the backbone returned.
            # If num_feature_levels == 1 then the backbone will only return the last one. Otherwise each is returned.
            src, mask = feat.decompose()
            srcs.append(self.input_proj[lvl](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            # If more feature levels are required than the backbone provides then additional feature maps are created
            _len_srcs = len(srcs)
            for lvl in range(_len_srcs, self.num_feature_levels):
                if lvl == _len_srcs:
                    src = self.input_proj[lvl](features[-1].tensors)
                else:
                    src = self.input_proj[lvl](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)  # embedding to determine in which feature level each query pixel lies in
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        if self.ref_points_mode == 'bbox':
            reference_points = pred_boxes[:, :, :2]  # predicted bbox centers from backbone
        else:
            reference_points = None

        if self.query_embedding_mode == 'learned':
            query_embeds = self.query_embed.weight

        # Pass everything to the transformer

        # srcs = feature maps from the backbone
        # pos = feature-level position embeddings (in which feature level does each query pixel (value) lie in)
        # reference_points = object queries = either bbox center coordinates from backbone or learned during training.
        #                    Specifies where attention module in Deformable DETR "searches" for objects).

        # hs = hidden states for each decoding layer
        hs, init_reference, _, _, _ = self.transformer(srcs, masks, pos, query_embeds, reference_points)
        
        # Extract mass information for force matrix prediction (if enabled)
        masses = None
        if self.use_force_prediction and targets is not None:
            masses = []
            for target in targets:
                # Get mass information
                if 'masses' in target:
                    target_masses = target['masses']
                else:
                    # Default mass if not available
                    target_masses = torch.ones(len(target['boxes']), device=hs.device)
                
                # Pad masses to match n_queries
                if len(target_masses) < self.n_queries:
                    padding = torch.zeros(self.n_queries - len(target_masses), device=hs.device)
                    target_masses = torch.cat([target_masses, padding])
                elif len(target_masses) > self.n_queries:
                    target_masses = target_masses[:self.n_queries]
                
                masses.append(target_masses)
            
            masses = torch.stack(masses)  # [bs, n_queries]
        elif self.use_force_prediction:
            # Use dummy values during inference
            bs = hs.shape[1]
            masses = torch.ones(bs, self.n_queries, device=hs.device)

        outputs_translation = []
        outputs_rotation = []
        outputs_force_matrices = []
        bs, _ = pred_classes.shape
        output_idx = torch.where(pred_classes > 0, pred_classes, 0).view(-1)

        # Iterate over the decoder outputs (hidden states) to calculate the intermediate and final outputs
        for lvl in range(hs.shape[0]):
            output_rotation = self.rotation_head[lvl](hs[lvl])  # (bs, n_queries, (n_classes + 1) * 6)
            output_translation = self.translation_head[lvl](hs[lvl])  # (bs, n_queries, (n_classes + 1) * 3)
            
            # Force matrix prediction
            if self.use_force_prediction:
                # Embed mass information
                mass_embeds = self.mass_embed(masses.unsqueeze(-1))  # [bs, n_queries, mass_embed_dim]
                
                # Predict pairwise force matrix using ForceMatrixHead
                output_force_matrix = self.force_matrix_head[lvl](hs[lvl], mass_embeds=mass_embeds)  # (bs, n_queries, n_queries, 3)
                outputs_force_matrices.append(output_force_matrix)
            
            # If class mode specific, select predicted rotation & translation according to predicted class from backbone
            if self.class_mode == 'specific':
                # reshape rotation output for later indexing by class
                output_rotation = output_rotation.view(bs * self.n_queries, self.n_classes, -1)
                # output_rotation = torch.cat([query[output_idx[i], :] for i, query in enumerate(output_rotation)]).view(
                #     bs, self.n_queries, -1)

                # Select the correct output according to the predicted class in the class-specific mode
                selected_rotations = []
                for i, query in enumerate(output_rotation):
                    # print(f"Query {i} output_idx: {output_idx[i]}, selected rotation: {query[output_idx[i], :]}")
                    selected_rotations.append(query[output_idx[i], :])
                output_rotation = torch.cat(selected_rotations).view(bs, self.n_queries, -1)  # (bs, n_queries, 6) => "6D representation of rotation"

                # reshape translation output for later indexing by class
                output_translation = output_translation.view(bs * self.n_queries, self.n_classes, -1)
                # output_translation = torch.cat(
                #     [query[output_idx[i], :] for i, query in enumerate(output_translation)]).view(bs, self.n_queries,
                #                                                                                   -1)

                # Select translation for the predicted class
                selected_translations = []
                for i, query in enumerate(output_translation):
                    # print(f"Query {i} output_idx: {output_idx[i]}, selected translation: {query[output_idx[i], :]}")
                    selected_translations.append(query[output_idx[i], :])
                output_translation = torch.cat(selected_translations).view(bs, self.n_queries, -1)  # (bs, n_queries, 3)

            # transform 6D rotation representation to 3x3 rotation matrix (or quaternion)
            output_rotation = self.process_rotation(output_rotation)

            outputs_rotation.append(output_rotation)
            outputs_translation.append(output_translation)

        outputs_rotation = torch.stack(outputs_rotation)
        outputs_translation = torch.stack(outputs_translation)
        
        # Stack force matrix outputs if available
        if self.use_force_prediction and len(outputs_force_matrices) > 0:
            outputs_force_matrices = torch.stack(outputs_force_matrices)

        # Predictions on the last hidden states (last decoder layer) of decoder are the final predictions
        out = {'pred_translation': outputs_translation[-1], 'pred_rotation': outputs_rotation[-1],
               'pred_boxes': pred_boxes, 'pred_classes': pred_classes}
        
        # Add force matrix predictions to output if available
        if self.use_force_prediction and len(outputs_force_matrices) > 0:
            # Apply inverse scaling to force predictions for inference
            # Note: During training, the criterion will handle scaling internally
            if not self.training:
                # During inference, apply inverse scaling to get actual force values
                force_scale_factor = getattr(self, 'force_scale_factor', 100.0)
                out['pred_force_matrix'] = outputs_force_matrices[-1] / force_scale_factor  # [bs, n_queries, n_queries, 3]
            else:
                # During training, keep scaled values for loss computation
                out['pred_force_matrix'] = outputs_force_matrices[-1]  # [bs, n_queries, n_queries, 3]

        # "aux_loss" are the intermediate predictions
        if self.aux_loss:
            if self.use_force_prediction and len(outputs_force_matrices) > 0:
                # Apply inverse scaling to auxiliary force outputs during inference
                if not self.training:
                    force_scale_factor = getattr(self, 'force_scale_factor', 100.0)
                    scaled_aux_force_matrices = outputs_force_matrices / force_scale_factor
                    out['aux_outputs'] = self._set_aux_loss(outputs_translation, outputs_rotation, pred_boxes, pred_classes, scaled_aux_force_matrices)
                else:
                    out['aux_outputs'] = self._set_aux_loss(outputs_translation, outputs_rotation, pred_boxes, pred_classes, outputs_force_matrices)
            else:
                out['aux_outputs'] = self._set_aux_loss(outputs_translation, outputs_rotation, pred_boxes, pred_classes)

        # n_boxes_per_sample = predicted bboxes per image
        return out, n_boxes_per_sample

    def _set_aux_loss(self, outputs_translation, outputs_quaternion, pred_boxes, pred_classes, outputs_force_matrices=None):
        if outputs_force_matrices is not None:
            return [{'pred_translation': t, 'pred_rotation': r, 'pred_boxes': pred_boxes, 'pred_classes': pred_classes, 'pred_force_matrix': f}
                    for t, r, f in zip(outputs_translation[:-1], outputs_quaternion[:-1], outputs_force_matrices[:-1])]
        else:
            return [{'pred_translation': t, 'pred_rotation': r, 'pred_boxes': pred_boxes, 'pred_classes': pred_classes}
                    for t, r in zip(outputs_translation[:-1], outputs_quaternion[:-1])]

    def process_rotation(self, pred_rotation):
        """
        Processes the predicted output rotation given the rotation mode.
        '6d' --> Gram Schmidt
        'quat' or 'silho_quat' --> L2 normalization
        else: Raise error
        """
        if self.rotation_mode == '6d':
            return self.rotation_6d_to_matrix(pred_rotation)
        elif self.rotation_mode in ['quat', 'silho_quat']:
            return F.normalize(pred_rotation, p=2, dim=2)
        else:
            raise NotImplementedError('Rotation mode is not supported')

    def rotation_6d_to_matrix(self, rot_6d):
        """
        Given a 6D rotation output, calculate the 3D rotation matrix in SO(3) using the Gramm Schmit process

        For details: https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.pdf
        """
        bs, n_q, _ = rot_6d.shape
        rot_6d = rot_6d.view(-1, 6)
        m1 = rot_6d[:, 0:3]
        m2 = rot_6d[:, 3:6]

        x = F.normalize(m1, p=2, dim=1)
        z = torch.cross(x, m2, dim=1)
        z = F.normalize(z, p=2, dim=1)
        y = torch.cross(z, x, dim=1)
        rot_matrix = torch.cat((x.view(-1, 3, 1), y.view(-1, 3, 1), z.view(-1, 3, 1)), 2)  # Rotation Matrix lying in the SO(3)
        rot_matrix = rot_matrix.view(bs, n_q, 3, 3)  #.transpose(2, 3)
        return rot_matrix


class SetCriterion(nn.Module):
    """ This class computes the loss for PoET, which consists of translation and rotation for now.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise translation and rotation)
    """
    def __init__(self, matcher, weight_dict, losses, hard_negative_ratio=0.2, force_scale_factor=100.0):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            hard_negative_ratio: ratio of hard negatives to keep for improved negative sampling
            force_scale_factor: scaling factor for force values to improve training stability
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.hard_negative_ratio = hard_negative_ratio
        self.force_scale_factor = force_scale_factor

    def loss_translation(self, outputs, targets, indices):
        """
        Compute the loss related to the translation of pose estimation, namely the mean square error (MSE).
        outputs must contain the key 'pred_translation', while targets must contain the key 'relative_position'
        Position / Translation are expected in [x, y, z] meters
        """
        idx = self._get_src_permutation_idx(indices)
        src_translation = outputs["pred_translation"][idx]
        
        # Check each target for relative_position
        valid_targets = []
        for batch_idx, (t, (_, i)) in enumerate(zip(targets, indices)):
            if 'relative_position' in t:
                if len(i) > 0:
                    selected = t['relative_position'][i]
                    valid_targets.append(selected)
                else:
                    print(f"[DEBUG] Batch {batch_idx}: Empty indices - no objects to match")
            else:
                print(f"[DEBUG] Batch {batch_idx}: No relative_position in target!")
        
        if not valid_targets:
            print("[DEBUG] NO VALID TARGETS FOUND! This explains the division by zero.")
            return {"loss_trans": torch.tensor(0.0, device=outputs["pred_translation"].device)}
        
        tgt_translation = torch.cat(valid_targets, dim=0)
        n_obj = len(tgt_translation)

        if n_obj == 0:
            return {"loss_trans": torch.tensor(0.0, device=outputs["pred_translation"].device)}

        loss_translation = F.mse_loss(src_translation, tgt_translation, reduction='none')
        loss_translation = torch.sum(loss_translation, dim=1)
        loss_translation = torch.sqrt(loss_translation)
        losses = {}
        losses["loss_trans"] = loss_translation.sum() / n_obj
        return losses

    def loss_rotation(self, outputs, targets, indices):
        """
        Compute the loss related to the rotation of pose estimation represented by a 3x3 rotation matrix.
        The function calculates the geodesic distance between the predicted and target rotation.
        L = arccos( 0.5 * (Trace(R\tilde(R)^T) -1)
        Calculates the loss in radiant.
        """
        eps = 1e-6
        idx = self._get_src_permutation_idx(indices)
        src_rot = outputs["pred_rotation"][idx]
        tgt_rot = torch.cat([t['relative_rotation'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        n_obj = len(tgt_rot)

        product = torch.bmm(src_rot, tgt_rot.transpose(1, 2))
        trace = torch.sum(product[:, torch.eye(3).bool()], 1)
        theta = torch.clamp(0.5 * (trace - 1), -1 + eps, 1 - eps)
        rad = torch.acos(theta)
        losses = {}
        losses["loss_rot"] = rad.sum() / n_obj
        return losses

    def loss_quaternion(self, outputs, targets, indices):
        """
        Compute the loss related to the rotation of pose estimation represented in quaternions, namely the quaternion loss
        Q_loss = - log(<q_pred,pred_gt>² + eps), where eps is a small values for stability reasons

        outputs must contain the key 'pred_quaternion', while targets must contain the key 'relative_quaternions'
        Quaternions expected in representation [w, x, y, z]
        """
        eps = 1e-4
        idx = self._get_src_permutation_idx(indices)
        src_quaternion = outputs["pred_rotation"][idx]
        tgt_quaternion = torch.cat([t['relative_quaternions'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        n_obj = len(tgt_quaternion)
        bs, q_dim = tgt_quaternion.shape

        dot_product = torch.mul(src_quaternion, tgt_quaternion)
        dp_sum = torch.sum(dot_product, 1)
        dp_square = torch.square(dp_sum)
        loss_quat = - torch.log(dp_square + eps)

        losses = {}
        losses["loss_rot"] = loss_quat.sum() / n_obj
        return losses

    def loss_silho_quaternion(self, outputs, targets, indices):
        """
        Compute the loss related to the rotation of pose estimation represented in quaternions, namely the quaternion loss
        Q_loss = log(1 - |<q_pred,pred_gt>| + eps), where eps is a small values for stability reasons

        outputs must contain the key 'pred_quaternion', while targets must contain the key 'relative_quaternions'
        Quaternions expected in representation [w, x, y, z]
        """
        eps = 1e-4
        idx = self._get_src_permutation_idx(indices)
        src_quaternion = outputs["pred_rotation"][idx]
        tgt_quaternion = torch.cat([t['relative_quaternions'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        n_obj = len(tgt_quaternion)
        bs, q_dim = tgt_quaternion.shape

        dot_product = torch.mul(src_quaternion, tgt_quaternion)
        dp_sum = torch.sum(dot_product, 1)
        loss_quat = torch.log(1 - torch.abs(dp_sum) + eps)

        losses = {}
        losses["loss_rot"] = loss_quat.sum() / n_obj
        return losses

    def loss_force_matrix(self, outputs, targets, indices):
        """
        Compute the loss related to pairwise force matrix prediction with improved negative sampling strategy.
        outputs must contain the key 'pred_force_matrix', while targets must contain the key 'force_matrix'
        """
        if 'pred_force_matrix' not in outputs:
            # If no force matrix predictions, return zero loss
            return {"loss_force_matrix": torch.tensor(0.0, device=next(iter(outputs.values())).device)}
        
        pred_force_matrices = outputs["pred_force_matrix"]  # [bs, n_queries, n_queries, 3]
        batch_size = pred_force_matrices.shape[0]
        n_queries = pred_force_matrices.shape[1]
        device = pred_force_matrices.device
        
        # Extract target force matrices for each batch
        target_force_matrices = []
        valid_batches = 0
        
        for i, (t, (src_idx, tgt_idx)) in enumerate(zip(targets, indices)):
            if 'force_matrix' in t and t['force_matrix'] is not None:
                valid_batches += 1
                target_matrix = t['force_matrix']  # [N, N, 3] where N is number of objects in this image
                
                # Pad or truncate to match n_queries
                n_objects = target_matrix.shape[0]
                
                if n_objects < n_queries:
                    # Pad with zeros
                    pad_size = n_queries - n_objects
                    target_matrix = F.pad(target_matrix, (0, 0, 0, pad_size, 0, pad_size), mode='constant', value=0)
                elif n_objects > n_queries:
                    # Truncate
                    target_matrix = target_matrix[:n_queries, :n_queries, :]
                
                # Reorder according to Hungarian matching indices
                reordered_matrix = torch.zeros_like(target_matrix)
                for s1, t1 in zip(src_idx, tgt_idx):
                    for s2, t2 in zip(src_idx, tgt_idx):
                        if s1 < n_queries and s2 < n_queries and t1 < target_matrix.shape[0] and t2 < target_matrix.shape[1]:
                            reordered_matrix[s1, s2] = target_matrix[t1, t2]
                
                target_force_matrices.append(reordered_matrix)
            else:
                # If no force matrix available for this batch, use zeros
                zero_matrix = torch.zeros(n_queries, n_queries, 3, device=device)
                target_force_matrices.append(zero_matrix)
        
        if not target_force_matrices:
            return {"loss_force_matrix": torch.tensor(0.0, device=device)}
        
        target_force_matrices = torch.stack(target_force_matrices)  # [bs, n_queries, n_queries, 3]
        
        # Apply force scaling to target values for improved training stability
        target_force_matrices = target_force_matrices * self.force_scale_factor
        
        total_loss = 0.0
        valid_loss_count = 0
        
        # Hard negative sampling parameters (similar to graph loss)
        hard_negative_ratio = getattr(self, 'hard_negative_ratio', 0.2)
        
        # Process each batch separately for improved negative sampling
        for batch_idx in range(batch_size):
            pred_matrix = pred_force_matrices[batch_idx]  # [n_queries, n_queries, 3]
            target_matrix = target_force_matrices[batch_idx]  # [n_queries, n_queries, 3]
            
            # Calculate force magnitude for positive/negative classification
            target_force_magnitude = torch.norm(target_matrix, dim=-1)  # [n_queries, n_queries]
            pred_force_magnitude = torch.norm(pred_matrix, dim=-1)  # [n_queries, n_queries]
            
            # Define positive and negative samples based on force magnitude threshold
            # Scale threshold according to force_scale_factor to maintain consistency
            force_threshold = 1e-2 * self.force_scale_factor  # Threshold for considering a force as existing
            pos_mask = (target_force_magnitude > force_threshold).float()
            neg_mask = (target_force_magnitude <= force_threshold).float()
            
            num_pos = pos_mask.sum()
            num_neg = neg_mask.sum()
            
            if num_pos == 0:
                # If no positive samples, skip this batch
                continue
            
            # Improved negative sampling strategy
            if num_pos > 0 and self.training and hard_negative_ratio < 1.0:
                with torch.no_grad():
                    # Calculate prediction confidence for negative samples
                    neg_confidence = pred_force_magnitude * neg_mask
                    # Select hard negatives (those with high predicted force but should be zero)
                    k = int(num_neg * hard_negative_ratio)
                    k = max(k, int(num_pos * 3))  # At least 3x positive samples
                    
                    if k < num_neg and k > 0:
                        hard_neg_values, _ = torch.topk(neg_confidence.flatten(), k)
                        if len(hard_neg_values) > 0:
                            neg_threshold = hard_neg_values[-1]
                            hard_neg_mask = (neg_confidence >= neg_threshold).float() * neg_mask
                            neg_mask = hard_neg_mask
            
            # Create combined sample mask
            sample_mask = pos_mask + neg_mask
            
            if sample_mask.sum() == 0:
                continue
            
            # Calculate weighted MSE loss for force vectors
            force_loss = F.mse_loss(pred_matrix, target_matrix, reduction='none')  # [n_queries, n_queries, 3]
            force_loss = force_loss.sum(dim=-1)  # Sum over force vector dimensions -> [n_queries, n_queries]
            
            # Apply positive weight balancing
            pos_weight = (neg_mask.sum() / (pos_mask.sum() + 1e-6))
            pos_weight = torch.clamp(pos_weight, min=1.0, max=10.0)
            
            # Weight positive samples more heavily
            weighted_loss = force_loss * sample_mask
            weighted_loss = torch.where(pos_mask.bool(), 
                                      weighted_loss * pos_weight, 
                                      weighted_loss)
            
            # Calculate batch loss
            if sample_mask.sum() > 0:
                batch_loss = weighted_loss.sum() / sample_mask.sum()
                total_loss += batch_loss
                valid_loss_count += 1
                
                # Optional: print debug information occasionally
                if self.training and torch.rand(1).item() < 0.001:  # 1% chance
                    print(f"Force Loss Debug - Batch {batch_idx}: pos_samples={int(num_pos)}, "
                          f"neg_samples={int(neg_mask.sum())}, pos_weight={pos_weight:.2f}, "
                          f"batch_loss={batch_loss:.6f}")
        
        if valid_loss_count > 0:
            total_loss = total_loss / valid_loss_count
        else:
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Ensure loss is not NaN
        if torch.isnan(total_loss):
            print("Warning: NaN detected in force matrix loss, replacing with 0.")
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        return {"loss_force_matrix": total_loss}

    def loss_force_symmetry(self, outputs, targets, indices):
        """
        Compute the normalized masked symmetry loss for force prediction.
        Enforces Newton's third law: F_ij = -F_ji for interacting object pairs.
        Uses normalization to balance constraints across different force magnitudes.
        """
        if 'pred_force_matrix' not in outputs:
            return {"loss_force_symmetry": torch.tensor(0.0, device=next(iter(outputs.values())).device)}
        
        pred_force_matrices = outputs["pred_force_matrix"]  # [bs, n_queries, n_queries, 3]
        batch_size = pred_force_matrices.shape[0]
        n_queries = pred_force_matrices.shape[1]
        device = pred_force_matrices.device
        
        total_symmetry_loss = 0.0
        valid_loss_count = 0
        
        for batch_idx in range(batch_size):
            pred_matrix = pred_force_matrices[batch_idx]  # [n_queries, n_queries, 3]
            
            # Get interaction mask from targets if available
            interaction_mask = torch.ones(n_queries, n_queries, device=device)
            
            # Extract interaction mask from target if available
            if batch_idx < len(targets):
                target = targets[batch_idx]
                if 'force_matrix' in target and target['force_matrix'] is not None:
                    target_matrix = target['force_matrix']
                    n_objects = target_matrix.shape[0]
                    
                    # Create interaction mask based on target force magnitudes
                    target_force_magnitude = torch.norm(target_matrix, dim=-1)
                    # Scale threshold according to force_scale_factor to maintain consistency
                    force_threshold = 1e-2 * self.force_scale_factor
                    target_interaction_mask = (target_force_magnitude > force_threshold).float()
                    
                    # Pad or truncate to match n_queries
                    if n_objects < n_queries:
                        pad_size = n_queries - n_objects
                        target_interaction_mask = F.pad(target_interaction_mask, (0, pad_size, 0, pad_size), mode='constant', value=0)
                    elif n_objects > n_queries:
                        target_interaction_mask = target_interaction_mask[:n_queries, :n_queries]
                    
                    interaction_mask = target_interaction_mask
            
            # Remove diagonal elements (self-interaction)
            diagonal_mask = torch.eye(n_queries, device=device)
            interaction_mask = interaction_mask * (1 - diagonal_mask)
            
            # Calculate force magnitudes for normalization
            pred_force_magnitude = torch.norm(pred_matrix, dim=-1)  # [n_queries, n_queries]
            
            # Create symmetric pairs mask (only consider upper triangle to avoid double counting)
            upper_triangle_mask = torch.triu(torch.ones(n_queries, n_queries, device=device), diagonal=1)
            symmetric_pairs_mask = interaction_mask * upper_triangle_mask
            
            if symmetric_pairs_mask.sum() == 0:
                continue
            
            # Calculate symmetry violation: F_ij + F_ji
            symmetry_violation = pred_matrix + pred_matrix.transpose(0, 1)  # [n_queries, n_queries, 3]
            symmetry_violation_magnitude = torch.norm(symmetry_violation, dim=-1)  # [n_queries, n_queries]
            
            # Normalization weights based on force magnitudes
            # Use the maximum magnitude of the pair for normalization
            force_mag_ij = pred_force_magnitude
            force_mag_ji = pred_force_magnitude.transpose(0, 1)
            max_force_magnitude = torch.maximum(force_mag_ij, force_mag_ji)
            
            # Avoid division by zero
            normalization_weights = 1.0 / (max_force_magnitude + 1e-6)
            
            # Apply normalization and masking
            normalized_symmetry_loss = symmetry_violation_magnitude * normalization_weights * symmetric_pairs_mask
            
            # 只对有意义的正样本进行平均 - 筛选强交互对
            # Scale threshold according to force_scale_factor to maintain consistency
            force_threshold = 1e-2 * self.force_scale_factor  # 力阈值，筛选强交互对
            strong_interaction_mask = (max_force_magnitude > force_threshold) * symmetric_pairs_mask
            
            # Calculate batch symmetry loss - 只对强交互对进行平均
            if strong_interaction_mask.sum() > 0:
                strong_normalized_loss = normalized_symmetry_loss * strong_interaction_mask
                batch_symmetry_loss = strong_normalized_loss.sum() / strong_interaction_mask.sum()
                total_symmetry_loss += batch_symmetry_loss
                valid_loss_count += 1
        
        if valid_loss_count > 0:
            total_symmetry_loss = total_symmetry_loss / valid_loss_count
        else:
            total_symmetry_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        return {"loss_force_symmetry": total_symmetry_loss}

    def loss_force_consistency(self, outputs, targets, indices):
        """
        Compute the physical consistency loss for force prediction.
        Enforces Newton's first law: sum of all forces on each object should be zero.
        """
        if 'pred_force_matrix' not in outputs:
            return {"loss_force_consistency": torch.tensor(0.0, device=next(iter(outputs.values())).device)}
        
        pred_force_matrices = outputs["pred_force_matrix"]  # [bs, n_queries, n_queries, 3]
        batch_size = pred_force_matrices.shape[0]
        n_queries = pred_force_matrices.shape[1]
        device = pred_force_matrices.device
        
        total_consistency_loss = 0.0
        valid_loss_count = 0
        
        for batch_idx in range(batch_size):
            pred_matrix = pred_force_matrices[batch_idx]  # [n_queries, n_queries, 3]
            
            # Get valid object mask from targets if available
            valid_objects_mask = torch.ones(n_queries, device=device)
            
            if batch_idx < len(targets):
                target = targets[batch_idx]
                src_idx, tgt_idx = indices[batch_idx]
                
                # Create mask for valid objects (those that are matched)
                valid_objects_mask = torch.zeros(n_queries, device=device)
                if len(src_idx) > 0:
                    valid_objects_mask[src_idx] = 1.0
            
            if valid_objects_mask.sum() == 0:
                continue
            
            # Calculate net force on each object
            # For object i, net force = sum_j F_ji (forces from all other objects j acting on i)
            # Note: F_ij is force from object i acting on object j
            net_forces = torch.sum(pred_matrix, dim=0)  # Sum over first dimension (forces acting on each object)
            
            # Calculate consistency loss (net force should be zero)
            net_force_magnitudes = torch.norm(net_forces, dim=-1)  # [n_queries]
            
            # Apply valid objects mask
            masked_net_force_magnitudes = net_force_magnitudes * valid_objects_mask
            
            # 只对有意义的正样本进行平均 - 筛选受到显著力作用的对象
            # Scale threshold according to force_scale_factor to maintain consistency
            net_force_threshold = 1e-2 * self.force_scale_factor  # 净力阈值，筛选受到显著力作用的对象
            significant_force_mask = (net_force_magnitudes > net_force_threshold) * valid_objects_mask
            
            # Calculate batch consistency loss - 只对受到显著力作用的对象进行平均
            if significant_force_mask.sum() > 0:
                significant_net_force_magnitudes = net_force_magnitudes * significant_force_mask
                batch_consistency_loss = significant_net_force_magnitudes.sum() / significant_force_mask.sum()
                total_consistency_loss += batch_consistency_loss
                valid_loss_count += 1
        
        if valid_loss_count > 0:
            total_consistency_loss = total_consistency_loss / valid_loss_count
        else:
            total_consistency_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        return {"loss_force_consistency": total_consistency_loss}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            'translation': self.loss_translation,
            'rotation': self.loss_rotation,
            'quaternion': self.loss_quaternion,
            'silho_quaternion': self.loss_silho_quaternion,
            'force_matrix': self.loss_force_matrix,
            'force_symmetry': self.loss_force_symmetry,
            'force_consistency': self.loss_force_consistency
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets, n_boxes):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            n_boxes: Number of predicted objects per image
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, n_boxes)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets, n_boxes)
                for loss in self.losses:
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            indices = self.matcher(enc_outputs, bin_targets, n_boxes)
            for loss in self.losses:
                kwargs = {}
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    model = PoET(
        backbone,
        transformer,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        n_classes=args.n_classes,
        bbox_mode=args.bbox_mode,
        ref_points_mode=args.reference_points,
        query_embedding_mode=args.query_embedding,
        rotation_mode=args.rotation_representation,
        class_mode=args.class_mode,
        aux_loss=args.aux_loss,
        backbone_type=args.backbone,
        use_graph_transformer=getattr(args, 'use_graph_transformer', True),
        graph_hidden_dim=getattr(args, 'graph_hidden_dim', None),
        graph_num_layers=getattr(args, 'graph_num_layers', 4),
        graph_num_heads=getattr(args, 'graph_num_heads', 4),
        use_force_prediction=getattr(args, 'use_force_prediction', True),
        force_scale_factor=getattr(args, 'force_scale_factor', 100.0)
    )

    matcher = build_matcher(args)
    weight_dict = {'loss_trans': args.translation_loss_coef, 'loss_rot': args.rotation_loss_coef}

    if args.rotation_representation == '6d':
        losses = ['translation', 'rotation']
    elif args.rotation_representation == 'quat':
        losses = ['translation', 'quaternion']
    elif args.rotation_representation == 'silho_quat':
        losses = ['translation', 'silho_quaternion']
    else:
        raise NotImplementedError('Rotation representation not implemented')
    
    # Add force matrix loss if force prediction is enabled
    if getattr(args, 'use_force_prediction', True):
        losses.append('force_matrix')
        losses.append('force_symmetry')
        losses.append('force_consistency')
        weight_dict['loss_force_matrix'] = getattr(args, 'force_loss_coef', 1.0)
        weight_dict['loss_force_symmetry'] = getattr(args, 'force_symmetry_coef', 0.5)
        weight_dict['loss_force_consistency'] = getattr(args, 'force_consistency_coef', 0.3)

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(matcher, weight_dict, losses, 
                             hard_negative_ratio=getattr(args, 'hard_negative_ratio', 0.2),
                             force_scale_factor=getattr(args, 'force_scale_factor', 100.0))
    criterion.to(device)

    return model, criterion, matcher
