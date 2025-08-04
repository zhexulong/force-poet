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

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
import time
import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy
import numpy as np
import torch
from matplotlib import pyplot as plt
from sympy.strategies.core import switch
from tabulate import tabulate

import util.logger
import util.misc as utils
from util.quaternion_ops import quat2rot
from data_utils.data_prefetcher import data_prefetcher

from evaluation_tools.metrics import get_src_permutation_idx, calc_rotation_error, calc_translation_error


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('position_loss', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('rotation_loss', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('force_loss', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # Mixed force loss components
    metric_logger.add_meter('mixed_force_loss', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('contact_loss', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('force_component_loss', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('contact_weight', utils.SmoothedValue(window_size=1, fmt='{value:.3f}'))
    metric_logger.add_meter('force_weight', utils.SmoothedValue(window_size=1, fmt='{value:.3f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):

        outputs, n_boxes_per_sample = model(samples, targets)
        loss_dict = criterion(outputs, targets, n_boxes_per_sample)
        weight_dict = criterion.weight_dict.copy()  # 创建副本避免修改原始权重
        
        # 如果epoch小于20，将所有force loss权重设置为0
        if epoch < 20:
            force_loss_keys = ['loss_force_matrix', 'loss_force_symmetry', 'loss_force_consistency']
            for force_key in force_loss_keys:
                if force_key in weight_dict:
                    weight_dict[force_key] = 0.0
                # 对所有auxiliary loss也设置为0
                for key in list(weight_dict.keys()):
                    if key.startswith(f'{force_key}_'):
                        weight_dict[key] = 0.0
        
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(position_loss=loss_dict_reduced['loss_trans'] * weight_dict['loss_trans'])
        metric_logger.update(rotation_loss=loss_dict_reduced['loss_rot'] * weight_dict['loss_rot'])
        if 'loss_force_matrix' in loss_dict_reduced:
            metric_logger.update(force_loss=loss_dict_reduced['loss_force_matrix'] * weight_dict['loss_force_matrix'])
        else:
            # 确保force_loss meter被初始化，即使没有force loss
            metric_logger.update(force_loss=0.0)
        
        # Update mixed force loss components if available
        if 'loss_mixed_force_contact' in loss_dict_reduced:
            metric_logger.update(mixed_force_loss=loss_dict_reduced['loss_mixed_force_contact'] * weight_dict.get('loss_mixed_force_contact', 1.0))
        else:
            metric_logger.update(mixed_force_loss=0.0)
            
        if 'loss_contact_component' in loss_dict_reduced:
            metric_logger.update(contact_loss=loss_dict_reduced['loss_contact_component'])
        else:
            metric_logger.update(contact_loss=0.0)
            
        if 'loss_force_component' in loss_dict_reduced:
            metric_logger.update(force_component_loss=loss_dict_reduced['loss_force_component'])
        else:
            metric_logger.update(force_component_loss=0.0)
            
        if 'contact_weight' in loss_dict_reduced:
            metric_logger.update(contact_weight=loss_dict_reduced['contact_weight'])
        else:
            metric_logger.update(contact_weight=0.0)
            
        if 'force_weight' in loss_dict_reduced:
            metric_logger.update(force_weight=loss_dict_reduced['force_weight'])
        else:
            metric_logger.update(force_weight=0.0)
            
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_with_iter_eval(model: torch.nn.Module, criterion: torch.nn.Module,
                                   data_loader: Iterable, optimizer: torch.optim.Optimizer,
                                   device: torch.device, epoch: int, global_iteration: int,
                                   eval_interval: int, eval_func, eval_args,
                                   max_norm: float = 0):
    """
    训练一个epoch，并在指定的迭代间隔进行评估
    
    Args:
        model: 模型
        criterion: 损失函数
        data_loader: 数据加载器
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
        global_iteration: 全局迭代计数器
        eval_interval: 评估间隔（迭代数）
        eval_func: 评估函数
        eval_args: 评估函数的参数
        max_norm: 梯度裁剪的最大范数
    
    Returns:
        tuple: (训练统计信息, 更新后的全局迭代计数器)
    """
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('position_loss', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('rotation_loss', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('force_loss', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for iter_idx in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        global_iteration += 1

        outputs, n_boxes_per_sample = model(samples, targets)
        loss_dict = criterion(outputs, targets, n_boxes_per_sample)
        weight_dict = criterion.weight_dict.copy()  # 创建副本避免修改原始权重
        
        # 如果epoch小于20，将所有force loss权重设置为0
        if epoch < 20:
            force_loss_keys = ['loss_force_matrix', 'loss_force_symmetry', 'loss_force_consistency']
            for force_key in force_loss_keys:
                if force_key in weight_dict:
                    weight_dict[force_key] = 0.0
                # 对所有auxiliary loss也设置为0
                for key in list(weight_dict.keys()):
                    if key.startswith(f'{force_key}_'):
                        weight_dict[key] = 0.0
        
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(position_loss=loss_dict_reduced['loss_trans'] * weight_dict['loss_trans'])
        metric_logger.update(rotation_loss=loss_dict_reduced['loss_rot'] * weight_dict['loss_rot'])
        if 'loss_force_matrix' in loss_dict_reduced:
            metric_logger.update(force_loss=loss_dict_reduced['loss_force_matrix'] * weight_dict['loss_force_matrix'])
        
        # Update mixed force loss components if available
        if 'loss_mixed_force_contact' in loss_dict_reduced:
            metric_logger.update(mixed_force_loss=loss_dict_reduced['loss_mixed_force_contact'] * weight_dict.get('loss_mixed_force_contact', 1.0))
        if 'loss_contact_component' in loss_dict_reduced:
            metric_logger.update(contact_loss=loss_dict_reduced['loss_contact_component'])
        if 'loss_force_component' in loss_dict_reduced:
            metric_logger.update(force_component_loss=loss_dict_reduced['loss_force_component'])
        if 'contact_weight' in loss_dict_reduced:
            metric_logger.update(contact_weight=loss_dict_reduced['contact_weight'])
        if 'force_weight' in loss_dict_reduced:
            metric_logger.update(force_weight=loss_dict_reduced['force_weight'])
            
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        # 检查是否需要进行评估
        if global_iteration % eval_interval == 0:
            print(f"\n=== 在第 {global_iteration} 次迭代后进行评估 ===")
            eval_func(**eval_args)
            model.train()  # 评估后重新设置为训练模式
            criterion.train()

        samples, targets = prefetcher.next()
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, global_iteration


# Function to convert normalized bounding box to un-normalized pixel values
def convert_to_pixel_values(img_width, img_height, bbox):
  cx, cy, w, h = bbox
  x_min = int((cx - w / 2) * img_width)
  y_min = int((cy - h / 2) * img_height)
  width = int(w * img_width)
  height = int(h * img_height)
  return x_min, y_min, width, height


# Function to visualize bounding boxes on a single image
def visualize_bounding_boxes(image, pred_bboxes, gt_bboxes, pred_classes, gt_classes, show_gt = False):
  img_height, img_width = image.shape[:2]

  # Draw predicted bounding boxes in red
  for idx, bbox in enumerate(pred_bboxes):
    x_min, y_min, width, height = convert_to_pixel_values(img_width, img_height, bbox)
    image = cv2.rectangle(image, (x_min, y_min), (x_min + width, y_min + height), (0, 255, 0), 1)
    text_position = (x_min, y_min + height + 15)  # Positioned below the bbox
    image = cv2.putText(image, f'pred: {pred_classes[idx]}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

  if show_gt:
      # Draw ground truth bounding boxes in green
      for idx, bbox in enumerate(gt_bboxes):
        x_min, y_min, width, height = convert_to_pixel_values(img_width, img_height, bbox)
        image = cv2.rectangle(image, (x_min, y_min), (x_min + width, y_min + height), (255, 0, 0), 1)
        text_position = (x_min, y_min - 10)  # Positioned above the bbox
        image = cv2.putText(image, f'gt: {gt_classes[idx]}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

  return image

@torch.no_grad()
def pose_evaluate(model, matcher, pose_evaluator, data_loader, image_set, bbox_mode, rotation_mode, device, output_dir, args, epoch=None):
    """
    Evaluate PoET on the whole dataset, calculate the evaluation metrics and store the final performance.
    """
    model.eval()
    matcher.eval()

    # Reset pose evaluator to be empty
    pose_evaluator.reset()

    type_str: str = "test" if pose_evaluator.testing else "eval"

    # Check whether the evaluation folder exists, otherwise create it
    if epoch is not None:
        output_eval_dir = os.path.join(output_dir, type_str + "_" + image_set + "_" + bbox_mode + "_" + str(epoch))
    else:
        output_eval_dir = os.path.join(output_dir, type_str + "_" + image_set + "_" + bbox_mode)
    Path(output_eval_dir).mkdir(parents=True, exist_ok=True)

    output_eval_dir += "/"

    if args:
        util.logger.saveArgs(output_eval_dir, args)

    pd = {int(i):0 for i,_ in pose_evaluator.classes_map.items()}
    gt = {int(i):0 for i,_ in pose_evaluator.classes_map.items()}

    if pose_evaluator.testing:
        print("Process test dataset:")
    else:
        print("Process validation dataset:")

    n_images = len(data_loader.dataset.ids)
    bs = data_loader.batch_size
    start_time = time.time()
    processed_images = 0
    for samples, targets in data_loader:
        batch_start_time = time.time()
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs, n_boxes_per_sample = model(samples, targets)  # bbox format: cxcywh (unnormalized)

        # Handle cases where there are no ground truth objects in the image
        if not any(t['labels'].numel() > 0 for t in targets):
            # If no ground truth objects, continue to the next batch
            continue

        # TODO: Refactor case of no prediction(s)
        if outputs is None:
            continue

        # # Visualize and save bounding boxes
        # for i in range(samples.tensors.shape[0]):
        #   image_tensor = samples.tensors[i]
        #   image_np = image_tensor.permute(1,2,0).contiguous().cpu().numpy()  # Convert from CxHxW to HxWxC format
        #   image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
        #
        #   # image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        #
        #   pred_bboxes = outputs["pred_boxes"][i]  # Get predicted bounding boxes for this image
        #   gt_bboxes = targets[i]["boxes"]  # Get ground truth bounding boxes for this image
        #
        #   pred_classes = outputs["pred_classes"][i]
        #   gt_classes = targets[i]["labels"]
        #
        #   # Visualize bounding boxes
        #   image_with_boxes = visualize_bounding_boxes(image_np, pred_bboxes, gt_bboxes, pred_classes, gt_classes)
        #
        #   # Display the image
        #   plt.figure(figsize=(6, 6))
        #   plt.imshow(image_with_boxes)
        #   plt.axis('off')
        #   # plt.show()
        #
        #   if not os.path.exists(os.path.join(output_eval_dir, "bbox/")):
        #     os.makedirs(os.path.join(output_eval_dir, "bbox/"))
        #   plt.savefig(os.path.join(output_eval_dir, "bbox/", f"{targets[i]['image_id'].item()}.png"), bbox_inches="tight")
        #   plt.close()

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Extract final predictions and store them
        indices = matcher(outputs_without_aux, targets, n_boxes_per_sample)
        idx = get_src_permutation_idx(indices)

        # # ------------------- Visualize MATCHED Bounding Boxes -------------------
        # for i in range(samples.tensors.shape[0]):
        #   image_tensor = samples.tensors[i]
        #   image_np = image_tensor.permute(1,2,0).contiguous().cpu().numpy()  # Convert from CxHxW to HxWxC format
        #   image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
        #
        #   matched_pred_bboxes = outputs_without_aux["pred_boxes"][i][indices[i][0]]  # Get MATCHED predicted bounding boxes for this image using indices
        #   gt_bboxes = targets[i]["boxes"][indices[i][1]] if len(indices[i][1]) > 0 else torch.empty(0, 4) # Get MATCHED ground truth bounding boxes for this image using indices, handle empty GT boxes
        #
        #   matched_pred_classes = outputs_without_aux["pred_classes"][i][indices[i][0]] # Get MATCHED predicted classes for this image using indices
        #   gt_classes = targets[i]["labels"][indices[i][1]] if len(indices[i][1]) > 0 else torch.empty(0, dtype=torch.int64) # Get MATCHED ground truth classes for this image using indices, handle empty GT classes
        #
        #
        #   # Visualize MATCHED bounding boxes
        #   image_with_matched_boxes = visualize_bounding_boxes(image_np, matched_pred_bboxes, gt_bboxes, matched_pred_classes, gt_classes, True) # Visualize matched bounding boxes
        #
        #   # Save the image with MATCHED bounding boxes
        #   plt.figure(figsize=(6, 6))
        #   plt.imshow(image_with_matched_boxes)
        #   plt.axis('off')
        #
        #   if not os.path.exists(os.path.join(output_eval_dir, "bbox/")): # Create directory for matched bbox visualizations
        #     os.makedirs(os.path.join(output_eval_dir, "bbox/"))
        #   plt.savefig(os.path.join(output_eval_dir, "bbox/", f"{targets[i]['image_id'].item()}_1.png"), bbox_inches="tight") # Save matched bbox visualization
        #   plt.close() # Close the figure to free memory

        pred_translations = outputs_without_aux["pred_translation"][idx].detach().cpu().numpy()
        pred_rotations = outputs_without_aux["pred_rotation"][idx].detach().cpu().numpy()
        
        # Extract force matrix predictions if available
        pred_force_matrix = None
        pred_contact_matrix = None
        if "pred_force_matrix" in outputs_without_aux:
            # Force matrix should NOT be indexed by idx since it represents pairwise interactions
            # Keep the full [bs, n_queries, n_queries, 3] shape
            pred_force_matrix = outputs_without_aux["pred_force_matrix"].detach().cpu().numpy()  # [bs, n_queries, n_queries, 3]
            print(f"DEBUG FORCE MATRIX EXTRACTION:")
            print(f"  pred_force_matrix shape: {pred_force_matrix.shape}")
            print(f"  pred_force_matrix magnitude - max: {np.linalg.norm(pred_force_matrix, axis=-1).max():.6f}, min: {np.linalg.norm(pred_force_matrix, axis=-1).min():.6f}, mean: {np.linalg.norm(pred_force_matrix, axis=-1).mean():.6f}")

        if "pred_contact_matrix" in outputs_without_aux:
            # Contact matrix should also NOT be indexed by idx for the same reason
            pred_contact_matrix = outputs_without_aux["pred_contact_matrix"].detach().cpu().numpy()  # [bs, n_queries, n_queries, 1]
            print(f"DEBUG CONTACT MATRIX EXTRACTION:")
            print(f"  pred_contact_matrix shape: {pred_contact_matrix.shape}")

        if rotation_mode in ['quat', 'silho_quat']:
            pred_rotations = quat2rot(pred_rotations)

        tgt_translations = torch.cat([t['relative_position'][i] for t, (_, i) in zip(targets, indices)], dim=0).detach().cpu().numpy()
        tgt_rotations = torch.cat([t['relative_rotation'][i] for t, (_, i) in zip(targets, indices)], dim=0).detach().cpu().numpy()
        
        # Extract target force matrices if available
        tgt_force_matrices = []
        for t, (_, i) in zip(targets, indices):
            if 'force_matrix' in t and t['force_matrix'] is not None:
                target_matrix = t['force_matrix'].detach().cpu().numpy()  # [N, N, 3]
                tgt_force_matrices.append(target_matrix)
            else:
                # Create empty matrix if no force matrix available
                n_objects = len(i)
                tgt_force_matrices.append(np.zeros((n_objects, n_objects, 3)))

        obj_classes_idx = torch.cat([t['labels'][i] for t, (_, i) in zip(targets, indices)], dim=0).detach().cpu().numpy()
        intrinsics = torch.cat([t['intrinsics'][i] for t, (_, i) in zip(targets, indices)], dim=0).detach().cpu().numpy()
        img_files = [data_loader.dataset.coco.loadImgs(t["image_id"].item())[0]['file_name'] for t, (_, i) in zip(targets, indices) for _ in range(0, len(i))]

        # Iterate over all predicted objects and save them in the pose evaluator
        for i, (cls_idx, img_file, intrinsic, pred_translation, pred_rotation, tgt_translation, tgt_rotation) in \
                enumerate(zip(obj_classes_idx, img_files, intrinsics, pred_translations, pred_rotations, tgt_translations, tgt_rotations)):
            # cls = pose_evaluator.classes[cls_idx - 1]
            cls = pose_evaluator.classes_map[str(cls_idx)]
            pose_evaluator.poses_pred[cls].append(
                np.concatenate((pred_rotation, pred_translation.reshape(3, 1)), axis=1))
            pose_evaluator.poses_gt[cls].append(
                np.concatenate((tgt_rotation, tgt_translation.reshape(3, 1)), axis=1))
            pose_evaluator.poses_img[cls].append(img_file)
            pose_evaluator.num[cls] += 1
            pose_evaluator.camera_intrinsics[cls].append(intrinsic)
        
        if pred_force_matrix is not None and tgt_force_matrices:
            # Group by image/batch for proper force matrix storage
            obj_idx_offset = 0  # Track position in obj_classes_idx array
            for batch_idx, (batch_target, (src_idx, tgt_idx)) in enumerate(zip(targets, indices)):
                if batch_idx >= len(tgt_force_matrices):
                    break
                    
                n_objects_in_batch = len(tgt_idx)
                if n_objects_in_batch == 0:
                    continue
                
                # Extract predicted and ground truth force matrices for this batch
                if batch_idx < pred_force_matrix.shape[0]:
                    # Get the force matrix for this batch - now correctly shaped [n_queries, n_queries, 3]
                    batch_pred_matrix = pred_force_matrix[batch_idx]  # [n_queries, n_queries, 3]
                    batch_gt_matrix = tgt_force_matrices[batch_idx]  # [N, N, 3]
                    
                    # Extract contact matrix predictions if available
                    batch_pred_contact = None
                    if pred_contact_matrix is not None:
                        batch_pred_contact = pred_contact_matrix[batch_idx]  # [n_queries, n_queries, 1]
                    
                    # For storage, we need to decide how to associate force matrices with classes
                    # Using the primary class for simplicity - store full matrix for primary class
                    if n_objects_in_batch > 0:
                        primary_cls_idx = obj_classes_idx[obj_idx_offset]
                        primary_cls = pose_evaluator.classes_map[str(primary_cls_idx)]
                        
                        # Apply matching logic: reorder predictions to match ground truth size
                        # Following the same logic as in loss functions
                        n_objects = batch_gt_matrix.shape[0]  # Ground truth size
                        n_queries = batch_pred_matrix.shape[0]  # Prediction size (both dimensions)
                        
                        # Convert batch_pred_matrix to tensor if it's numpy array
                        if isinstance(batch_pred_matrix, np.ndarray):
                            batch_pred_matrix = torch.from_numpy(batch_pred_matrix).to(device)
                        
                        # Reorder predictions using Hungarian matching indices
                        reordered_pred_matrix = torch.zeros((n_objects, n_objects, 3), 
                                                           dtype=batch_pred_matrix.dtype, 
                                                           device=batch_pred_matrix.device)
                        
                        # Map predictions to ground truth positions using indices
                        for i, tgt_i in enumerate(tgt_idx):
                            for j, tgt_j in enumerate(tgt_idx):  # Use tgt_idx for both dimensions
                                if i < len(src_idx) and j < len(src_idx):
                                    src_i = src_idx[i]
                                    src_j = src_idx[j]
                                    if src_i < n_queries and src_j < n_queries:
                                        reordered_pred_matrix[tgt_i, tgt_j] = batch_pred_matrix[src_i, src_j]
                        
                        # Convert to numpy for storage
                        if hasattr(reordered_pred_matrix, 'cpu'):
                            final_pred_matrix = reordered_pred_matrix.cpu().numpy()
                        else:
                            final_pred_matrix = reordered_pred_matrix
                        
                        # Debug: Print force matrix information
                        print(f"DEBUG Force Matrix Reordering:")
                        print(f"  batch_pred_matrix shape: {batch_pred_matrix.shape}")
                        print(f"  final_pred_matrix shape: {final_pred_matrix.shape}")
                        print(f"  gt_matrix_final shape: {batch_gt_matrix.shape}")
                        print(f"  tgt_idx: {tgt_idx}")
                        print(f"  src_idx: {src_idx}")
                        
                        # Check original predicted force magnitudes before reordering
                        if hasattr(batch_pred_matrix, 'cpu'):
                            orig_pred_matrix = batch_pred_matrix.cpu().numpy()
                        else:
                            orig_pred_matrix = batch_pred_matrix
                        orig_pred_magnitude = np.linalg.norm(orig_pred_matrix, axis=-1)
                        print(f"  ORIGINAL batch_pred_matrix magnitude - max: {orig_pred_magnitude.max():.6f}, min: {orig_pred_magnitude.min():.6f}, mean: {orig_pred_magnitude.mean():.6f}")
                        print(f"  ORIGINAL > 1e-2: {(orig_pred_magnitude > 1e-2).sum()}, > 1e-1: {(orig_pred_magnitude > 1e-1).sum()}, > 1.0: {(orig_pred_magnitude > 1.0).sum()}")
                        
                        # Check predicted force magnitudes after reordering
                        pred_force_magnitude = np.linalg.norm(final_pred_matrix, axis=-1)
                        print(f"  REORDERED pred_force_magnitude - max: {pred_force_magnitude.max():.6f}, min: {pred_force_magnitude.min():.6f}, mean: {pred_force_magnitude.mean():.6f}")
                        print(f"  REORDERED > 1e-2: {(pred_force_magnitude > 1e-2).sum()}, > 1e-1: {(pred_force_magnitude > 1e-1).sum()}, > 1.0: {(pred_force_magnitude > 1.0).sum()}")
                        
                        # Check ground truth force magnitudes
                        gt_force_magnitude = np.linalg.norm(batch_gt_matrix, axis=-1)
                        print(f"  GT force magnitude - max: {gt_force_magnitude.max():.6f}, min: {gt_force_magnitude.min():.6f}, mean: {gt_force_magnitude.mean():.6f}")
                        print(f"  GT > 1e-2: {(gt_force_magnitude > 1e-2).sum()}, > 1e-1: {(gt_force_magnitude > 1e-1).sum()}, > 1.0: {(gt_force_magnitude > 1.0).sum()}")
                        
                        
                        # Process contact matrix predictions if available
                        final_pred_contact = None
                        if batch_pred_contact is not None:
                            # Convert batch_pred_contact to tensor if it's numpy array
                            if isinstance(batch_pred_contact, np.ndarray):
                                batch_pred_contact = torch.from_numpy(batch_pred_contact).to(device)
                            
                            # Reorder contact predictions using the same matching logic
                            reordered_pred_contact = torch.zeros((n_objects, n_objects, 1), 
                                                                dtype=batch_pred_contact.dtype, 
                                                                device=batch_pred_contact.device)
                            
                            # Map contact predictions to ground truth positions using indices
                            for i, tgt_i in enumerate(tgt_idx):
                                for j, tgt_j in enumerate(tgt_idx):  # Use tgt_idx for both dimensions
                                    if i < len(src_idx) and j < len(src_idx) and i < n_queries and j < n_queries:
                                        # Extract the contact value properly, handling different tensor shapes
                                        contact_value = batch_pred_contact[i, j]
                                        if contact_value.dim() > 1:
                                            contact_value = contact_value.squeeze()
                                        reordered_pred_contact[tgt_i, tgt_j] = contact_value
                            
                            # Convert to numpy for storage
                            if hasattr(reordered_pred_contact, 'cpu'):
                                final_pred_contact = reordered_pred_contact.cpu().numpy()
                            else:
                                final_pred_contact = reordered_pred_contact
                        
                        # Use ground truth matrix as-is (no resizing needed)
                        gt_matrix_final = batch_gt_matrix.copy()
                        
                        # Create ground truth contact matrix from force matrix
                        # Contact is determined by force magnitude > threshold
                        contact_threshold = 1e-2  # Can be made configurable

                        gt_force_magnitude = np.linalg.norm(gt_matrix_final, axis=-1)  # [n_objects, n_objects]

                        gt_contact_matrix = (gt_force_magnitude > contact_threshold).astype(np.float32)  # [n_objects, n_objects]

                        # Store the force matrices
                        pose_evaluator.force_matrices_pred[primary_cls].append(final_pred_matrix)
                        pose_evaluator.force_matrices_gt[primary_cls].append(gt_matrix_final)
                        
                        # Store the contact matrices only if predictions are available and valid
                        if final_pred_contact is not None:
                            pose_evaluator.contact_matrices_pred[primary_cls].append(final_pred_contact)
                            pose_evaluator.contact_matrices_gt[primary_cls].append(gt_contact_matrix)
                
                # Update obj_idx_offset to track position in obj_classes_idx array
                obj_idx_offset += n_objects_in_batch

        batch_total_time = time.time() - batch_start_time
        batch_total_time_str = str(datetime.timedelta(seconds=int(batch_total_time)))
        processed_images = processed_images + len(targets)
        remaining_images = n_images - processed_images
        remaining_batches = remaining_images / bs
        eta = batch_total_time * remaining_batches
        eta_str = str(datetime.timedelta(seconds=int(eta)))
        print("Processed {}/{} \t Batch Time: {} \t ETA: {}".format(processed_images, n_images, batch_total_time_str, eta_str))
    # At this point iterated over all validation images and for each object the result is fed into the pose evaluator
    total_time = time.time() - start_time
    time_per_img = total_time / n_images
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    time_per_img_str = str(datetime.timedelta(seconds=int(time_per_img)))
    print("Network Processing Time\nTotal Time: {}\t\tImages: {}\t\ts/img: {}".format(total_time_str, n_images, time_per_img_str))
    print("Start results evaluation")
    start_time = time.time()
    print("Start Calculating ADD")
    pose_evaluator.evaluate_pose_add(output_eval_dir)
    print("Start Calculating ADD-S")
    pose_evaluator.evaluate_pose_adi(output_eval_dir)
    print("Start Calculating ADD(-S)")
    pose_evaluator.evaluate_pose_adds(output_eval_dir)
    print("Start Calculating Average Translation Error")
    avg_trans = pose_evaluator.calculate_class_avg_translation_error(output_eval_dir, epoch)
    print("Start Calculating Average Rotation Error")
    avg_rot = pose_evaluator.calculate_class_avg_rotation_error(output_eval_dir, epoch)
    
    # Add force evaluation if force predictions are available
    avg_force = None
    force_mae = None
    force_rmse = None
    force_matrix_results = None
    avg_force_matrix_error = None
    
    has_force_data = any(len(pose_evaluator.forces_pred[cls]) > 0 for cls in pose_evaluator.classes)
    has_force_matrix_data = any(len(pose_evaluator.force_matrices_pred[cls]) > 0 for cls in pose_evaluator.classes)
    
    # Traditional force evaluation (if simple force data exists)
    if has_force_data:
        print("Start Calculating Force Prediction Metrics")
        force_mae, force_rmse = pose_evaluator.evaluate_force_prediction(output_eval_dir, epoch)
        print("Start Calculating Average Force Error")
        avg_force = pose_evaluator.calculate_class_avg_force_error(output_eval_dir, epoch)
        print(f"Force Evaluation Results:")
        print(f"  Average Force Error: {avg_force:.6f}")
        print(f"  Force MAE: {force_mae:.6f}")
        print(f"  Force RMSE: {force_rmse:.6f}")
    
    # Force matrix evaluation (detailed force interaction analysis)
    if has_force_matrix_data:
        print("Start Calculating Detailed Force Matrix Metrics")
        force_matrix_results = pose_evaluator.evaluate_force_matrix_prediction(output_eval_dir, epoch)
        print("Start Calculating Average Force Matrix Error")
        avg_force_matrix_error = pose_evaluator.calculate_class_avg_force_matrix_error(output_eval_dir, epoch)
        
        # Print detailed force matrix results
        if force_matrix_results and "overall" in force_matrix_results:
            overall_results = force_matrix_results["overall"]
            print(f"Force Matrix Evaluation Results:")
            print(f"  Average Force Matrix Error: {avg_force_matrix_error:.6f}")
            print(f"  Vector MSE: {overall_results['vector_mse']:.6f}")
            print(f"  Vector MAE: {overall_results['vector_mae']:.6f}")
            print(f"  Direction Accuracy: {overall_results['direction_accuracy']:.6f}")
            print(f"  Detection Accuracy: {overall_results['detection_accuracy']:.6f}")
            print(f"  Precision: {overall_results['precision']:.6f}")
            print(f"  Recall: {overall_results['recall']:.6f}")
            print(f"  F1 Score: {overall_results['f1_score']:.6f}")
    
    # Contact classification and conditional force regression evaluation
    contact_classification_results = None
    conditional_force_results = None
    
    # Check if contact classification data is available
    has_contact_data = any(len(pose_evaluator.contact_matrices_pred[cls]) > 0 for cls in pose_evaluator.classes)
    
    if has_contact_data:
        print("Start Calculating Contact Classification Metrics")
        contact_classification_results = pose_evaluator.evaluate_contact_classification(output_eval_dir, epoch)
        
        if contact_classification_results and "overall" in contact_classification_results:
            overall_contact = contact_classification_results["overall"]
            print(f"Contact Classification Evaluation Results:")
            print(f"  Overall Accuracy: {overall_contact['accuracy']:.6f}")
            print(f"  Overall Precision: {overall_contact['precision']:.6f}")
            print(f"  Overall Recall: {overall_contact['recall']:.6f}")
            print(f"  Overall F1 Score: {overall_contact['f1_score']:.6f}")
            print(f"  Overall AUC: {overall_contact['auc']:.6f}")
        
        print("Start Calculating Conditional Force Regression Metrics")
        conditional_force_results = pose_evaluator.evaluate_conditional_force_regression(output_eval_dir, epoch)
        
        if conditional_force_results and "overall" in conditional_force_results:
            overall_cond_force = conditional_force_results["overall"]
            print(f"Conditional Force Regression Evaluation Results:")
            print(f"  Contact Region MSE: {overall_cond_force['contact_mse']:.6f}")
            print(f"  Contact Region MAE: {overall_cond_force['contact_mae']:.6f}")
            print(f"  Non-Contact Region MSE: {overall_cond_force['noncontact_mse']:.6f}")
            print(f"  Non-Contact Region MAE: {overall_cond_force['noncontact_mae']:.6f}")
            print(f"  Overall MSE: {overall_cond_force['overall_mse']:.6f}")
            print(f"  Overall MAE: {overall_cond_force['overall_mae']:.6f}")
    
    if not has_force_data and not has_force_matrix_data:
        print("No force data available for evaluation")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Evaluation time: {}".format(total_time_str))
    
    # Return evaluation results
    eval_results = {
        'avg_trans': avg_trans,
        'avg_rot': avg_rot,
        'avg_force': avg_force,
        'force_mae': force_mae,
        'force_rmse': force_rmse,
        'avg_force_matrix_error': avg_force_matrix_error,
        'force_matrix_results': force_matrix_results,
        'contact_classification_results': contact_classification_results,
        'conditional_force_results': conditional_force_results
    }
    return eval_results


@torch.no_grad()
def bop_evaluate(model, matcher, data_loader, image_set, bbox_mode, rotation_mode, device, output_dir):
    """
    Evaluate PoET on the dataset and store the results in the BOP format
    """
    model.eval()
    matcher.eval()

    output_eval_dir = output_dir + "/bop_" + bbox_mode + "/"
    Path(output_eval_dir).mkdir(parents=True, exist_ok=True)

    out_csv_file = open(output_eval_dir + 'ycbv.csv', 'w')
    out_csv_file.write("scene_id,im_id,obj_id,score,R,t,time")
    n_images = len(data_loader.dataset.ids)

    # CSV format: scene_id, im_id, obj_id, score, R, t, time
    counter = 1
    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        pred_start_time = time.time()
        outputs, n_boxes_per_sample = model(samples, targets)
        pred_end_time = time.time() - pred_start_time
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        indices = matcher(outputs_without_aux, targets, n_boxes_per_sample)
        idx = get_src_permutation_idx(indices)

        pred_translations = outputs_without_aux["pred_translation"][idx].detach().cpu().numpy()
        pred_rotations = outputs_without_aux["pred_rotation"][idx].detach().cpu().numpy()

        if rotation_mode in ['quat', 'silho_quat']:
            pred_rotations = quat2rot(pred_rotations)

        obj_classes_idx = torch.cat([t['labels'][i] for t, (_, i) in zip(targets, indices)],
                                    dim=0).detach().cpu().numpy()

        img_files = [data_loader.dataset.coco.loadImgs(t["image_id"].item())[0]['file_name'] for t, (_, i) in
                     zip(targets, indices) for _ in range(0, len(i))]

        for cls_idx, img_file, pred_translation, pred_rotation in zip(obj_classes_idx, img_files, pred_translations, pred_rotations):
            file_info = img_file.split("/")
            scene_id = int(file_info[1])
            img_id = int(file_info[3][:file_info[3].rfind(".")])
            obj_id = cls_idx
            score = 1.0
            csv_str = "{},{},{},{},{} {} {} {} {} {} {} {} {}, {} {} {}, {}\n".format(scene_id, img_id, obj_id, score,
                                                                                    pred_rotation[0, 0], pred_rotation[0, 1], pred_rotation[0, 2],
                                                                                    pred_rotation[1, 0], pred_rotation[1, 1], pred_rotation[1, 2],
                                                                                    pred_rotation[2, 0], pred_rotation[2, 1], pred_rotation[2, 2],
                                                                                    pred_translation[0] * 1000, pred_translation[1] * 1000, pred_translation[2] * 1000,
                                                                                    pred_end_time)
            out_csv_file.write(csv_str)
        print("Processed {}/{}".format(counter, n_images))
        counter += 1

    out_csv_file.close()