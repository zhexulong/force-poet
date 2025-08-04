#!/usr/bin/env python3
"""
阈值分析评估脚本：计算不同阈值下的Precision、Recall和F1-Score，并绘制P-R曲线
"""

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import seaborn as sns

# 添加项目路径
sys.path.append('/data/gplong/force_map_project/w-poet/poet')

import util.misc as utils
import data_utils.samplers as samplers
from data_utils import build_dataset
from models import build_model
from evaluation_tools.pose_evaluator import PoseEvaluator
from evaluation_tools.metrics import get_src_permutation_idx
from util.quaternion_ops import quat2rot


def convert_numpy_types(obj):
    """
    递归地将numpy类型转换为Python原生类型，用于JSON序列化
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def get_args_parser():
    parser = argparse.ArgumentParser('PoET Threshold Analysis', add_help=False)
    
    # Model parameters
    parser.add_argument('--backbone', default='maskrcnn', type=str)
    parser.add_argument('--backbone_cfg', default='configs/isaac_sim.yaml', type=str)
    parser.add_argument('--backbone_weights', default=None, type=str)
    parser.add_argument('--position_embedding', default='sine', type=str)
    parser.add_argument('--nheads', default=16, type=int)
    parser.add_argument('--enc_layers', default=5, type=int)
    parser.add_argument('--dec_layers', default=5, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--num_queries', default=100, type=int)
    parser.add_argument('--num_feature_levels', default=4, type=int)
    parser.add_argument('--reference_points', default='learned', type=str)
    parser.add_argument('--query_embedding', default='learned', type=str)
    parser.add_argument('--class_mode', default='specific', type=str)
    parser.add_argument('--aux_loss', action='store_true')
    parser.add_argument('--translation_loss_coef', default=5.0, type=float)
    parser.add_argument('--rotation_loss_coef', default=1.0, type=float)
    
    # Training parameters
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--clip_max_norm', default=5.0, type=float)
    
    # Backbone parameters
    parser.add_argument('--backbone_conf_thresh', default=0.4, type=float)
    parser.add_argument('--backbone_iou_thresh', default=0.5, type=float)
    parser.add_argument('--backbone_agnostic_nms', action='store_true')
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding_scale', default=2 * 3.14159, type=float)
    
    # Transformer parameters
    parser.add_argument('--dim_feedforward', default=1024, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    
    # Matcher parameters
    parser.add_argument('--matcher_type', default='hungarian', type=str)
    parser.add_argument('--set_cost_class', default=2, type=float)
    parser.add_argument('--set_cost_bbox', default=5, type=float)
    parser.add_argument('--set_cost_giou', default=2, type=float)
    parser.add_argument('--set_cost_translation', default=5, type=float)
    parser.add_argument('--set_cost_rotation', default=1, type=float)
    
    # Dataset parameters
    parser.add_argument('--dataset', default='custom', type=str)
    parser.add_argument('--dataset_path', default='../isaac_sim_poet_dataset_new', type=str)
    parser.add_argument('--n_classes', default=21, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--eval_set', default='test_all', type=str)
    
    # Evaluation parameters
    parser.add_argument('--checkpoint_path', required=True, type=str, help='Path to model checkpoint')
    parser.add_argument('--output_dir', default='./results/threshold_analysis', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--bbox_mode', default='gt', type=str)
    parser.add_argument('--rotation_representation', default='6d', type=str)
    
    # Threshold analysis parameters
    parser.add_argument('--min_threshold', default=0.1, type=float, help='Minimum threshold for analysis')
    parser.add_argument('--max_threshold', default=0.9, type=float, help='Maximum threshold for analysis')
    parser.add_argument('--threshold_step', default=0.05, type=float, help='Step size for threshold analysis')
    
    # Force prediction parameters
    parser.add_argument('--use_force_prediction', default=True, type=bool)
    parser.add_argument('--class_info', default='/annotations/classes.json', type=str)
    parser.add_argument('--model_symmetry', default='/annotations/isaac_sim_symmetries.json', type=str)
    
    # Graph Transformer parameters (must match training configuration)
    parser.add_argument('--use_graph_transformer', action='store_true', default=True,
                        help="Whether to use graph transformer for force prediction")
    parser.add_argument('--graph_hidden_dim', default=256, type=int,
                        help="Hidden dimension for graph transformer (must match training)")
    parser.add_argument('--graph_num_layers', default=4, type=int,
                        help="Number of layers in graph transformer")
    parser.add_argument('--graph_num_heads', default=4, type=int,
                        help="Number of attention heads in graph transformer")
    parser.add_argument('--force_loss_coef', default=2.0, type=float,
                        help='Loss weighing parameter for the force prediction')
    parser.add_argument('--force_symmetry_coef', default=0.5, type=float,
                        help='Loss weighing parameter for force symmetry constraint')
    parser.add_argument('--force_consistency_coef', default=0.5, type=float,
                        help='Loss weighing parameter for force consistency constraint')
    parser.add_argument('--force_scale_factor', default=5.0, type=float,
                        help='Scaling factor for force values during training')
    
    # 添加缺失的参数
    parser.add_argument('--synt_background', default=None, type=str, help='使用合成背景')
    parser.add_argument('--rgb_augmentation', action='store_true', help='使用RGB增强')
    parser.add_argument('--no_augmentation', action='store_true', help='不使用数据增强')
    parser.add_argument('--grayscale', action='store_true', help='使用灰度图像')
    parser.add_argument('--jitter_probability', default=0.5, type=float, help='抖动概率')
    parser.add_argument('--cache_mode', action='store_true', help='启用缓存模式')
    
    return parser


@torch.no_grad()
def evaluate_with_thresholds(model, matcher, pose_evaluator, data_loader, device, args):
    """
    在不同阈值下评估模型性能，收集接触分类和力回归的预测与真值
    """
    model.eval()
    matcher.eval()
    pose_evaluator.reset()
    
    # 定义接触阈值
    contact_threshold = 1e-2
    
    print("Starting to collect prediction data...")
    print(f"Dataset has {len(data_loader)} batches in total")
    
    # 收集接触分类数据
    all_contact_probs = []
    all_contact_labels = []
    
    # 收集力回归数据（仅在有接触的点上）
    all_force_predictions = []  # 预测的力向量
    all_force_ground_truth = []  # 真值力向量
    all_contact_mask = []  # 接触掩码，用于条件回归评估
    
    for batch_idx, (samples, targets) in enumerate(data_loader):
        # 限制只处理前50个样本进行调试
        if batch_idx >= 50:
            print(f"Debug mode: stopping after 50 batches")
            break
            
        if batch_idx % 10 == 0:
            print(f"Processing batch {batch_idx + 1}/{min(50, len(data_loader))}...")
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        outputs, n_boxes_per_sample = model(samples, targets)
        
        if outputs is None:
            continue
            
        if not any(t['labels'].numel() > 0 for t in targets):
            continue
            
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        
        # 获取matcher的indices
        if hasattr(matcher, 'bbox_mode'):
            # PoseMatcher
            indices = matcher(outputs_without_aux, targets, n_boxes_per_sample)
        else:
            # HungarianMatcher - need to convert pred_classes to pred_logits format
            outputs_for_hungarian = outputs_without_aux.copy()
            
            # Validate and clean bounding boxes
            if 'pred_boxes' in outputs_for_hungarian:
                pred_boxes = outputs_for_hungarian['pred_boxes']
                invalid_mask = (pred_boxes == -1).any(dim=-1) | torch.isnan(pred_boxes).any(dim=-1) | torch.isinf(pred_boxes).any(dim=-1)
                dummy_boxes = torch.tensor([0.5, 0.5, 0.1, 0.1], device=pred_boxes.device).expand_as(pred_boxes)
                pred_boxes = torch.where(invalid_mask.unsqueeze(-1), dummy_boxes, pred_boxes)
                pred_boxes = torch.clamp(pred_boxes, 0.0, 1.0)
                outputs_for_hungarian['pred_boxes'] = pred_boxes
            
            if 'pred_classes' in outputs_for_hungarian:
                pred_classes = outputs_for_hungarian['pred_classes']
                batch_size, num_queries = pred_classes.shape
                num_classes = pred_classes.max().item() + 1 if pred_classes.numel() > 0 else 1
                
                pred_logits = torch.zeros(batch_size, num_queries, num_classes, device=pred_classes.device)
                for b in range(batch_size):
                    for q in range(num_queries):
                        class_idx = pred_classes[b, q].long()
                        if class_idx >= 0:
                            pred_logits[b, q, class_idx] = 10.0
                        else:
                            pred_logits[b, q, 0] = 10.0
                
                outputs_for_hungarian['pred_logits'] = pred_logits
                
            indices = matcher(outputs_for_hungarian, targets)
        
        # 提取接触预测和力预测
        if "pred_contact_matrix" in outputs_without_aux and "pred_force_matrix" in outputs_without_aux:
            pred_contact_logits = outputs_without_aux["pred_contact_matrix"]
            pred_force_matrix = outputs_without_aux["pred_force_matrix"]

            if batch_idx % 20 == 0:
                print(f"\n--- Batch {batch_idx+1} Debug Info ---")
                print(f"pred_contact_logits shape: {pred_contact_logits.shape}")
                print(f"pred_force_matrix shape: {pred_force_matrix.shape}")

            # 处理每个样本 - 参考engine.py的实现
            for i, (t, (src_idx, tgt_idx)) in enumerate(zip(targets, indices)):
                if 'force_matrix' in t and t['force_matrix'] is not None and len(tgt_idx) > 0:
                    # 获取目标force matrix
                    target_matrix = t['force_matrix'].detach().cpu().numpy()
                    
                    # 获取当前batch的预测
                    batch_pred_contact_logits = pred_contact_logits[i].detach().cpu().numpy()
                    batch_pred_force_matrix = pred_force_matrix[i].detach().cpu().numpy()
                    
                    n_matched = len(tgt_idx)
                    
                    if n_matched > 0:
                        # 参考engine.py的重排序逻辑
                        n_objects = target_matrix.shape[0]  # Ground truth size
                        n_queries = batch_pred_force_matrix.shape[0]  # Prediction size
                        
                        # 重新排序预测矩阵以匹配ground truth的大小和顺序
                        reordered_pred_force_matrix = np.zeros((n_objects, n_objects, 3), dtype=np.float32)
                        reordered_pred_contact_logits = np.zeros((n_objects, n_objects, 1), dtype=np.float32)
                        
                        # 使用indices映射预测到ground truth位置
                        for idx_i, tgt_i in enumerate(tgt_idx):
                            for idx_j, tgt_j in enumerate(tgt_idx):
                                if idx_i < len(src_idx) and idx_j < len(src_idx):
                                    src_i = src_idx[idx_i]
                                    src_j = src_idx[idx_j]
                                    if src_i < n_queries and src_j < n_queries:
                                        reordered_pred_force_matrix[tgt_i, tgt_j] = batch_pred_force_matrix[src_i, src_j]
                                        # 处理contact logits的维度
                                        if batch_pred_contact_logits.ndim == 3:
                                            reordered_pred_contact_logits[tgt_i, tgt_j] = batch_pred_contact_logits[src_i, src_j]
                                        else:
                                            reordered_pred_contact_logits[tgt_i, tgt_j, 0] = batch_pred_contact_logits[src_i, src_j]
                        
                        # 使用重排序后的矩阵
                        matched_pred_force_matrix = reordered_pred_force_matrix
                        matched_pred_contact_logits = reordered_pred_contact_logits
                        matched_target_matrix = target_matrix  # 直接使用原始target matrix
                        
                        # 创建接触标签（基于力的大小）- 调整阈值以提高precision
                        gt_force_magnitude = np.linalg.norm(matched_target_matrix, axis=-1)
                        gt_contact_matrix = (gt_force_magnitude > contact_threshold).astype(bool)  # 使用bool类型
                        
                        # 转换接触预测为概率
                        if matched_pred_contact_logits.ndim == 3 and matched_pred_contact_logits.shape[-1] == 1:
                            # 如果是 [N, N, 1] 格式，去掉最后一维
                            pred_contact_logits_2d = matched_pred_contact_logits.squeeze(-1)
                        else:
                            pred_contact_logits_2d = matched_pred_contact_logits
                        
                        pred_contact_probs = torch.sigmoid(torch.tensor(pred_contact_logits_2d)).numpy()
                        
                        # 添加详细的调试信息
                        if i == 0 and batch_idx < 5:  # 只在前5个batch的第一个样本打印详细信息
                            print(f"\n=== DEBUG INFO for batch {batch_idx}, sample {i} ===")
                            print(f"Ground truth force matrix shape: {matched_target_matrix.shape}")
                            print(f"Predicted contact logits shape: {matched_pred_contact_logits.shape}")
                            print(f"Force magnitude stats: min={gt_force_magnitude.min():.6f}, max={gt_force_magnitude.max():.6f}, mean={gt_force_magnitude.mean():.6f}")
                            print(f"Contact threshold: {contact_threshold}")
                            print(f"GT contact ratio: {gt_contact_matrix.mean():.4f} ({np.sum(gt_contact_matrix)}/{gt_contact_matrix.size})")
                            print(f"Predicted contact logits stats: min={pred_contact_logits_2d.min():.6f}, max={pred_contact_logits_2d.max():.6f}, mean={pred_contact_logits_2d.mean():.6f}")
                            print(f"Predicted contact probs stats: min={pred_contact_probs.min():.6f}, max={pred_contact_probs.max():.6f}, mean={pred_contact_probs.mean():.6f}")
                            
                            # 分析预测概率分布
                            high_prob_count = np.sum(pred_contact_probs > 0.5)
                            medium_prob_count = np.sum((pred_contact_probs > 0.1) & (pred_contact_probs <= 0.5))
                            low_prob_count = np.sum(pred_contact_probs <= 0.1)
                            print(f"Predicted prob distribution: >0.5: {high_prob_count}, 0.1-0.5: {medium_prob_count}, <=0.1: {low_prob_count}")
                            
                            # 分析真值中有接触的位置的预测概率
                            if np.sum(gt_contact_matrix) > 0:
                                contact_positions_probs = pred_contact_probs[gt_contact_matrix]
                                print(f"Predicted probs at true contact positions: min={contact_positions_probs.min():.6f}, max={contact_positions_probs.max():.6f}, mean={contact_positions_probs.mean():.6f}")
                                high_conf_contacts = np.sum(contact_positions_probs > 0.5)
                                print(f"True contacts with high confidence (>0.5): {high_conf_contacts}/{len(contact_positions_probs)} ({high_conf_contacts/len(contact_positions_probs)*100:.1f}%)")
                            
                            # 分析真值中无接触的位置的预测概率
                            if np.sum(~gt_contact_matrix) > 0:
                                no_contact_positions_probs = pred_contact_probs[~gt_contact_matrix]
                                print(f"Predicted probs at true non-contact positions: min={no_contact_positions_probs.min():.6f}, max={no_contact_positions_probs.max():.6f}, mean={no_contact_positions_probs.mean():.6f}")
                                false_positive_count = np.sum(no_contact_positions_probs > 0.5)
                                print(f"False positives (non-contacts with >0.5 prob): {false_positive_count}/{len(no_contact_positions_probs)} ({false_positive_count/len(no_contact_positions_probs)*100:.1f}%)")
                        
                        # 确保数据维度正确：NxN -> N*N
                        pred_contact_flat = pred_contact_probs.flatten()  # [N*N]
                        gt_contact_flat = gt_contact_matrix.flatten().astype(bool)  # [N*N] bool类型
                        
                        all_contact_probs.extend(pred_contact_flat)
                        all_contact_labels.extend(gt_contact_flat)
                        
                        # 收集力回归数据 - 确保是float类型的NxNx3
                        pred_force_flat = matched_pred_force_matrix.reshape(-1, 3).astype(np.float32)  # [N*N, 3]
                        gt_force_flat = matched_target_matrix.reshape(-1, 3).astype(np.float32)  # [N*N, 3]
                        contact_mask_flat = gt_contact_flat  # 接触掩码 [N*N] bool类型
                        
                        all_force_predictions.extend(pred_force_flat)
                        all_force_ground_truth.extend(gt_force_flat)
                        all_contact_mask.extend(contact_mask_flat)
                        
                        if i == 0 and batch_idx % 20 == 0:
                            print(f"Matched shapes - contact: {matched_pred_contact_logits.shape}, force: {matched_pred_force_matrix.shape}")
                            print(f"Contact positive ratio: {gt_contact_matrix.mean():.4f}")
                            print(f"Force magnitude range: [{gt_force_magnitude.min():.6f}, {gt_force_magnitude.max():.6f}]")
    
    print(f"Collected {len(all_contact_probs)} contact prediction samples")
    print(f"Collected {len(all_force_predictions)} force prediction samples")
    
    # 转换为numpy数组，确保正确的数据类型
    all_contact_probs = np.array(all_contact_probs, dtype=np.float32)  # [N*N] float32
    all_contact_labels = np.array(all_contact_labels, dtype=bool)      # [N*N] bool
    all_force_predictions = np.array(all_force_predictions, dtype=np.float32)  # [N*N, 3] float32
    all_force_ground_truth = np.array(all_force_ground_truth, dtype=np.float32) # [N*N, 3] float32
    all_contact_mask = np.array(all_contact_mask, dtype=bool)          # [N*N] bool
    
    # 数据分布调试信息
    print(f"\n=== Data distribution debugging info ===")
    print(f"Contact prediction probability range: [{all_contact_probs.min():.6f}, {all_contact_probs.max():.6f}]")
    print(f"Contact positive samples: {np.sum(all_contact_labels)} / {len(all_contact_labels)} ({np.mean(all_contact_labels.astype(float)):.4f})")
    print(f"Force prediction range: [{all_force_predictions.min():.6f}, {all_force_predictions.max():.6f}]")
    print(f"Force ground truth range: [{all_force_ground_truth.min():.6f}, {all_force_ground_truth.max():.6f}]")
    print(f"Contact points for force evaluation: {np.sum(all_contact_mask)}")
    print(f"Data shapes - contact_probs: {all_contact_probs.shape}, contact_labels: {all_contact_labels.shape}")
    print(f"Data shapes - force_pred: {all_force_predictions.shape}, force_gt: {all_force_ground_truth.shape}")
    print(f"Data types - contact_probs: {all_contact_probs.dtype}, contact_labels: {all_contact_labels.dtype}")
    print(f"Data types - force_pred: {all_force_predictions.dtype}, force_gt: {all_force_ground_truth.dtype}")
    print(f"Contact threshold used: {contact_threshold}")
    
    # 详细分析预测概率分布
    print(f"\n=== Detailed prediction analysis ===")
    prob_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i in range(len(prob_bins)-1):
        count = np.sum((all_contact_probs >= prob_bins[i]) & (all_contact_probs < prob_bins[i+1]))
        print(f"Predictions in [{prob_bins[i]:.1f}, {prob_bins[i+1]:.1f}): {count} ({count/len(all_contact_probs)*100:.1f}%)")
    
    # 分析真值为正样本时的预测分布
    positive_mask = all_contact_labels == True
    if np.sum(positive_mask) > 0:
        positive_probs = all_contact_probs[positive_mask]
        print(f"\n=== True positive samples prediction analysis ===")
        print(f"True positive samples: {len(positive_probs)}")
        print(f"Prediction stats for true positives: min={positive_probs.min():.6f}, max={positive_probs.max():.6f}, mean={positive_probs.mean():.6f}, std={positive_probs.std():.6f}")
        for i in range(len(prob_bins)-1):
            count = np.sum((positive_probs >= prob_bins[i]) & (positive_probs < prob_bins[i+1]))
            print(f"True positives in [{prob_bins[i]:.1f}, {prob_bins[i+1]:.1f}): {count} ({count/len(positive_probs)*100:.1f}%)")
    
    # 分析真值为负样本时的预测分布
    negative_mask = all_contact_labels == False
    if np.sum(negative_mask) > 0:
        negative_probs = all_contact_probs[negative_mask]
        print(f"\n=== True negative samples prediction analysis ===")
        print(f"True negative samples: {len(negative_probs)}")
        print(f"Prediction stats for true negatives: min={negative_probs.min():.6f}, max={negative_probs.max():.6f}, mean={negative_probs.mean():.6f}, std={negative_probs.std():.6f}")
        for i in range(len(prob_bins)-1):
            count = np.sum((negative_probs >= prob_bins[i]) & (negative_probs < prob_bins[i+1]))
            print(f"True negatives in [{prob_bins[i]:.1f}, {prob_bins[i+1]:.1f}): {count} ({count/len(negative_probs)*100:.1f}%)")
    
    # 计算不同阈值下的precision预览
    print(f"\n=== Precision preview at different thresholds ===")
    test_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for thresh in test_thresholds:
        pred_binary = (all_contact_probs >= thresh).astype(bool)
        tp = np.sum((pred_binary == True) & (all_contact_labels == True))
        fp = np.sum((pred_binary == True) & (all_contact_labels == False))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (np.sum(all_contact_labels) + 1e-8)
        print(f"Threshold {thresh:.1f}: Precision={precision:.4f}, Recall={recall:.4f}, TP={tp}, FP={fp}")
    
    return {
        'contact_probs': all_contact_probs,
        'contact_labels': all_contact_labels,
        'force_predictions': all_force_predictions,
        'force_ground_truth': all_force_ground_truth,
        'contact_mask': all_contact_mask
    }


def evaluate_contact_classification(contact_probs, contact_labels, threshold=0.5):
    """
    评估接触点分类性能
    
    Args:
        contact_probs: 预测的接触概率 [N*N] float32
        contact_labels: 真值接触标签 [N*N] bool
        threshold: 分类阈值
    
    Returns:
        dict: 包含分类指标的字典
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
    
    # 转换概率为二分类预测
    predictions = (contact_probs >= threshold).astype(bool)
    
    # 转换bool标签为int用于sklearn计算
    contact_labels_int = contact_labels.astype(int)
    predictions_int = predictions.astype(int)
    
    # 计算分类指标
    precision = precision_score(contact_labels_int, predictions_int, zero_division=0)
    recall = recall_score(contact_labels_int, predictions_int, zero_division=0)
    f1 = f1_score(contact_labels_int, predictions_int, zero_division=0)
    accuracy = accuracy_score(contact_labels_int, predictions_int)
    
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(contact_labels_int, predictions_int).ravel()
    
    results = {
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'total_samples': len(contact_labels),
        'positive_samples': int(np.sum(contact_labels)),
        'negative_samples': int(np.sum(~contact_labels))
    }
    
    print(f"\n=== Contact Classification Results (threshold={threshold:.3f}) ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"True Positives: {tp}, True Negatives: {tn}")
    print(f"False Positives: {fp}, False Negatives: {fn}")
    print(f"Total samples: {len(contact_labels)}, Positive: {np.sum(contact_labels)}, Negative: {np.sum(~contact_labels)}")
    
    return results


def evaluate_conditional_force_regression(force_predictions, force_ground_truth, contact_mask):
    """
    评估条件力回归性能（仅在有接触的点上）
    
    Args:
        force_predictions: 预测的力向量 [N*N, 3] float32
        force_ground_truth: 真值力向量 [N*N, 3] float32
        contact_mask: 接触掩码 [N*N] bool，True表示有接触，False表示无接触
    
    Returns:
        dict: 包含回归指标的字典
    """
    # 仅选择有接触的点
    contact_indices = contact_mask == True
    
    if np.sum(contact_indices) == 0:
        print("Warning: No contact points found for force regression evaluation")
        return {
            'mae': float('inf'),
            'rmse': float('inf'),
            'mae_per_axis': [float('inf')] * 3,
            'rmse_per_axis': [float('inf')] * 3,
            'contact_points': 0,
            'total_points': len(contact_mask)
        }
    
    contact_pred_forces = force_predictions[contact_indices]
    contact_gt_forces = force_ground_truth[contact_indices]
    
    # 计算整体误差指标
    mae = np.mean(np.abs(contact_pred_forces - contact_gt_forces))
    mse = np.mean((contact_pred_forces - contact_gt_forces) ** 2)
    rmse = np.sqrt(mse)
    
    # 计算每个轴的误差
    mae_per_axis = np.mean(np.abs(contact_pred_forces - contact_gt_forces), axis=0)
    rmse_per_axis = np.sqrt(np.mean((contact_pred_forces - contact_gt_forces) ** 2, axis=0))
    
    # 计算力大小的误差
    pred_force_magnitude = np.linalg.norm(contact_pred_forces, axis=1)
    gt_force_magnitude = np.linalg.norm(contact_gt_forces, axis=1)
    magnitude_mae = np.mean(np.abs(pred_force_magnitude - gt_force_magnitude))
    magnitude_rmse = np.sqrt(np.mean((pred_force_magnitude - gt_force_magnitude) ** 2))
    
    # 计算方向误差（余弦相似度）
    # 避免零向量导致的数值问题
    pred_norm = np.linalg.norm(contact_pred_forces, axis=1)
    gt_norm = np.linalg.norm(contact_gt_forces, axis=1)
    valid_mask = (pred_norm > 1e-6) & (gt_norm > 1e-6)
    
    if np.sum(valid_mask) > 0:
        pred_normalized = contact_pred_forces[valid_mask] / pred_norm[valid_mask, np.newaxis]
        gt_normalized = contact_gt_forces[valid_mask] / gt_norm[valid_mask, np.newaxis]
        cosine_similarity = np.sum(pred_normalized * gt_normalized, axis=1)
        cosine_similarity = np.clip(cosine_similarity, -1, 1)  # 确保在有效范围内
        mean_cosine_similarity = np.mean(cosine_similarity)
        angular_error_rad = np.arccos(np.abs(cosine_similarity))
        mean_angular_error_deg = np.mean(angular_error_rad) * 180 / np.pi
    else:
        mean_cosine_similarity = 0.0
        mean_angular_error_deg = 90.0
    
    results = {
        'mae': mae,
        'rmse': rmse,
        'mae_per_axis': mae_per_axis.tolist(),
        'rmse_per_axis': rmse_per_axis.tolist(),
        'magnitude_mae': magnitude_mae,
        'magnitude_rmse': magnitude_rmse,
        'mean_cosine_similarity': mean_cosine_similarity,
        'mean_angular_error_deg': mean_angular_error_deg,
        'contact_points': np.sum(contact_indices),
        'total_points': len(contact_mask),
        'valid_direction_points': np.sum(valid_mask) if np.sum(valid_mask) > 0 else 0
    }
    
    print(f"\n=== Conditional Force Regression Results ===")
    print(f"Contact points used: {np.sum(contact_indices)} / {len(contact_mask)}")
    print(f"Overall MAE: {mae:.6f}")
    print(f"Overall RMSE: {rmse:.6f}")
    print(f"MAE per axis (X,Y,Z): [{mae_per_axis[0]:.6f}, {mae_per_axis[1]:.6f}, {mae_per_axis[2]:.6f}]")
    print(f"RMSE per axis (X,Y,Z): [{rmse_per_axis[0]:.6f}, {rmse_per_axis[1]:.6f}, {rmse_per_axis[2]:.6f}]")
    print(f"Force magnitude MAE: {magnitude_mae:.6f}")
    print(f"Force magnitude RMSE: {magnitude_rmse:.6f}")
    print(f"Mean cosine similarity: {mean_cosine_similarity:.4f}")
    print(f"Mean angular error: {mean_angular_error_deg:.2f} degrees")
    
    return results


def analyze_thresholds(contact_probs, contact_labels, args):
    """
    Analyze performance metrics at different thresholds
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    
    thresholds = np.arange(args.min_threshold, args.max_threshold + args.threshold_step, args.threshold_step)
    
    results = {
        'thresholds': [],
        'precisions': [],
        'recalls': [],
        'f1_scores': [],
        'accuracies': []
    }
    
    print("\nAnalyzing performance at different thresholds...")
    print(f"{'Threshold':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Accuracy':<10}")
    print("-" * 55)
    
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        # Convert probabilities to binary predictions
        pred_binary = (contact_probs > threshold).astype(int)
        
        # Calculate metrics using sklearn
        precision = precision_score(contact_labels, pred_binary, zero_division=0)
        recall = recall_score(contact_labels, pred_binary, zero_division=0)
        f1 = f1_score(contact_labels, pred_binary, zero_division=0)
        accuracy = accuracy_score(contact_labels, pred_binary)
        
        results['thresholds'].append(threshold)
        results['precisions'].append(precision)
        results['recalls'].append(recall)
        results['f1_scores'].append(f1)
        results['accuracies'].append(accuracy)
        
        print(f"{threshold:<10.2f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {accuracy:<10.4f}")
        
        # Record best F1 score and corresponding threshold
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\nBest threshold: {best_threshold:.2f} (F1-Score: {best_f1:.4f})")
    
    return results, best_threshold, best_f1


def plot_pr_curve(contact_probs, contact_labels, results, output_dir):
    """
    Plot Precision-Recall curves and threshold analysis
    """
    # Calculate P-R curve
    precision, recall, thresholds_pr = precision_recall_curve(contact_labels, contact_probs)
    average_precision = average_precision_score(contact_labels, contact_probs)
    
    # Set plot style
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. P-R curve
    ax1.plot(recall, precision, 'b-', linewidth=2, label=f'P-R Curve (AP={average_precision:.3f})')
    ax1.fill_between(recall, precision, alpha=0.3)
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # 2. Threshold vs Metrics
    ax2.plot(results['thresholds'], results['precisions'], 'r-', marker='o', label='Precision', linewidth=2)
    ax2.plot(results['thresholds'], results['recalls'], 'g-', marker='s', label='Recall', linewidth=2)
    ax2.plot(results['thresholds'], results['f1_scores'], 'b-', marker='^', label='F1-Score', linewidth=2)
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Metric Value')
    ax2.set_title('Performance Metrics vs Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # 3. F1-Score curve
    best_idx = np.argmax(results['f1_scores'])
    ax3.plot(results['thresholds'], results['f1_scores'], 'purple', linewidth=3)
    ax3.scatter(results['thresholds'][best_idx], results['f1_scores'][best_idx], 
               color='red', s=100, zorder=5, label=f'Best Threshold: {results["thresholds"][best_idx]:.2f}')
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('F1-Score')
    ax3.set_title('F1-Score vs Threshold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # 4. Accuracy curve
    ax4.plot(results['thresholds'], results['accuracies'], 'orange', linewidth=2, marker='d')
    ax4.set_xlabel('Threshold')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Accuracy vs Threshold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Save figure
    pr_curve_path = os.path.join(output_dir, 'precision_recall_analysis.png')
    plt.savefig(pr_curve_path, dpi=300, bbox_inches='tight')
    print(f"P-R curve saved to: {pr_curve_path}")
    
    return average_precision


def save_results(results, best_threshold, best_f1, average_precision, output_dir):
    """
    Save analysis results
    """
    # Save detailed results
    results_data = {
        'analysis_summary': {
            'best_threshold': float(best_threshold),
            'best_f1_score': float(best_f1),
            'average_precision': float(average_precision),
            'analysis_date': datetime.datetime.now().isoformat()
        },
        'threshold_analysis': {
            'thresholds': [float(t) for t in results['thresholds']],
            'precisions': [float(p) for p in results['precisions']],
            'recalls': [float(r) for r in results['recalls']],
            'f1_scores': [float(f) for f in results['f1_scores']],
            'accuracies': [float(a) for a in results['accuracies']]
        }
    }
    
    # Save JSON file
    json_path = os.path.join(output_dir, 'threshold_analysis_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Save CSV file
    csv_path = os.path.join(output_dir, 'threshold_analysis_results.csv')
    with open(csv_path, 'w') as f:
        f.write('Threshold,Precision,Recall,F1_Score,Accuracy\n')
        for i in range(len(results['thresholds'])):
            f.write(f"{results['thresholds'][i]:.3f},{results['precisions'][i]:.6f},"
                   f"{results['recalls'][i]:.6f},{results['f1_scores'][i]:.6f},"
                   f"{results['accuracies'][i]:.6f}\n")
    
    print(f"Results saved to:")
    print(f"  JSON: {json_path}")
    print(f"  CSV: {csv_path}")


def build_pose_evaluator(args):
    """构建姿态评估器"""
    # 创建简单的评估器用于测试
    # 这里使用简化的参数，实际使用时需要根据具体数据集调整
    models = {}
    classes = {str(i): f'class_{i}' for i in range(1, args.n_classes + 1)}
    model_info = {f'class_{i}': {'diameter': 0.1} for i in range(1, args.n_classes + 1)}
    model_symmetry = {f'class_{i}': False for i in range(1, args.n_classes + 1)}
    
    return PoseEvaluator(models, classes, model_info, model_symmetry)


def main(args):
    # 设置设备
    device = torch.device(args.device)
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"开始阈值分析评估...")
    print(f"检查点路径: {args.checkpoint_path}")
    print(f"输出目录: {args.output_dir}")
    
    # 构建数据集
    dataset_val = build_dataset(image_set=args.eval_set, args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                drop_last=False, collate_fn=utils.collate_fn, num_workers=2)
    
    # 构建模型
    model, criterion, matcher = build_model(args)
    model.to(device)
    
    # 加载模型检查点
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"加载检查点: {args.checkpoint_path}")
        try:
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            
            # 检查检查点文件的键
            print(f"检查点文件包含的键: {list(checkpoint.keys())}")
            
            # 尝试不同的键名
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'], strict=False)
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                # 如果没有找到标准键，尝试直接加载
                model.load_state_dict(checkpoint, strict=False)
            
            print(f"已加载检查点: {args.checkpoint_path}")
        except Exception as e:
            print(f"警告: 无法加载检查点 {args.checkpoint_path}: {e}")
            print("继续使用随机初始化的模型进行测试...")
    else:
        print("未提供检查点路径或文件不存在，使用随机初始化的模型进行测试...")
    
    # 构建评估器
    pose_evaluator = build_pose_evaluator(args)
    
    # 收集预测数据
    eval_data = evaluate_with_thresholds(
        model, matcher, pose_evaluator, data_loader_val, device, args
    )
    
    # 提取数据
    all_contact_probs = eval_data['contact_probs']
    all_contact_labels = eval_data['contact_labels']
    all_force_predictions = eval_data['force_predictions']
    all_force_ground_truth = eval_data['force_ground_truth']
    all_contact_mask = eval_data['contact_mask']
    
    if len(all_contact_probs) == 0:
        print("错误: 没有收集到有效的预测数据")
        return
    
    print(f"\n=== 开始综合评估 ===")
    
    # 1. 接触点分类评估
    print(f"\n1. 接触点分类评估")
    classification_results = evaluate_contact_classification(
        all_contact_probs, all_contact_labels, threshold=0.5
    )
    
    # 2. 条件力回归评估
    print(f"\n2. 条件力回归评估")
    regression_results = evaluate_conditional_force_regression(
        all_force_predictions, all_force_ground_truth, all_contact_mask
    )
    
    # 3. 阈值分析
    print(f"\n3. 阈值分析")
    results, best_threshold, best_f1 = analyze_thresholds(all_contact_probs, all_contact_labels, args)
    
    # 绘制P-R曲线
    average_precision = plot_pr_curve(all_contact_probs, all_contact_labels, results, args.output_dir)
    
    # 保存综合结果
    comprehensive_results = {
        'contact_classification': classification_results,
        'conditional_force_regression': regression_results,
        'threshold_analysis': {
            'thresholds': [float(t) for t in results['thresholds']],
            'precisions': [float(p) for p in results['precisions']],
            'recalls': [float(r) for r in results['recalls']],
            'f1_scores': [float(f) for f in results['f1_scores']],
            'accuracies': [float(a) for a in results['accuracies']],
            'best_threshold': float(best_threshold),
            'best_f1_score': float(best_f1),
            'average_precision': float(average_precision)
        },
        'data_summary': {
            'total_contact_samples': len(all_contact_probs),
            'total_force_samples': len(all_force_predictions),
            'contact_points': int(np.sum(all_contact_mask == 1)),
            'positive_contact_ratio': float(np.mean(all_contact_labels)),
            'contact_prob_range': [float(all_contact_probs.min()), float(all_contact_probs.max())],
            'force_pred_range': [float(all_force_predictions.min()), float(all_force_predictions.max())],
            'force_gt_range': [float(all_force_ground_truth.min()), float(all_force_ground_truth.max())]
        }
    }
    
    # 保存结果 - 转换numpy类型以支持JSON序列化
    results_file = os.path.join(args.output_dir, 'comprehensive_evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(convert_numpy_types(comprehensive_results), f, indent=2)
    
    save_results(results, best_threshold, best_f1, average_precision, args.output_dir)
    
    print(f"\n=== 评估总结 ===")
    print(f"接触分类F1分数: {classification_results['f1_score']:.4f}")
    print(f"接触分类准确率: {classification_results['accuracy']:.4f}")
    print(f"力回归MAE: {regression_results['mae']:.6f}")
    print(f"力回归RMSE: {regression_results['rmse']:.6f}")
    print(f"力方向余弦相似度: {regression_results['mean_cosine_similarity']:.4f}")
    print(f"推荐阈值: {best_threshold:.2f}")
    print(f"最佳F1分数: {best_f1:.4f}")
    print(f"平均精确率 (AP): {average_precision:.4f}")
    print(f"\n综合结果保存在: {results_file}")
    print(f"阈值分析结果保存在: {args.output_dir}")
    print("综合评估完成!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PoET阈值分析评估脚本', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)