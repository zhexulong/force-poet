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
import cv2
import random
from scipy.spatial.transform import Rotation as R

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
    parser.add_argument('--num_queries', default=20, type=int)
    parser.add_argument('--num_feature_levels', default=4, type=int)
    parser.add_argument('--reference_points', default='bbox', type=str)
    parser.add_argument('--query_embedding', default='bbox', type=str)
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
    parser.add_argument('--matcher_type', default='pose', type=str)
    parser.add_argument('--set_cost_class', default=2, type=float)
    parser.add_argument('--set_cost_bbox', default=5, type=float)
    parser.add_argument('--set_cost_giou', default=2, type=float)
    parser.add_argument('--set_cost_translation', default=5, type=float)
    parser.add_argument('--set_cost_rotation', default=1, type=float)
    
    # Dataset parameters
    parser.add_argument('--dataset', default='custom', type=str)
    parser.add_argument('--dataset_path', default='../isaac_sim_poet_dataset_new', type=str)
    parser.add_argument('--n_classes', default=22, type=int)
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
    
    # Visualization parameters
    parser.add_argument('--enable_visualization', action='store_true', default=True,
                        help='Enable force and pose visualization (default: True)')
    parser.add_argument('--viz_sample_rate', default=0.01, type=float,
                        help='Sample rate for visualization (default: 1%%)')
    parser.add_argument('--viz_output_dir', default='./results/visualizations', type=str,
                        help='Output directory for visualization images')
    parser.add_argument('--viz_force_scale', default=0.02, type=float,
                        help='Scale factor for force vector visualization')
    parser.add_argument('--viz_min_force', default=0.01, type=float,
                        help='Minimum force magnitude to visualize')
    
    return parser


def draw_forces_on_image(rgb_image, force_matrix_pred, contact_matrix_pred,  
                        force_matrix_gt, poses_6d_cam, camera_intrinsics, 
                        scale_factor=0.02, min_force_magnitude=0.01, obj_ids=None):
    """
    在RGB图像上绘制预测和真值的力向量
    
    Args:
        rgb_image: RGB图像 (H, W, 3)
        force_matrix_pred: 预测的力矩阵 (N, N, 3)
        contact_matrix_pred: 预测的接触矩阵 (N, N, 1)
        force_matrix_gt: 真值的力矩阵 (N, N, 3) 
        poses_6d_cam: 物体6D位姿 (N, 7) [x, y, z, qw, qx, qy, qz]
        camera_intrinsics: 相机内参矩阵 (3, 3)
        scale_factor: 力向量缩放因子
        min_force_magnitude: 最小力大小阈值
        obj_ids: 物体ID列表
        
    Returns:
        绘制了力向量的RGB图像
    """
    img = rgb_image.copy()
    
    if force_matrix_pred is None or force_matrix_gt is None or poses_6d_cam is None:
        return img
    
    n_objects = min(len(poses_6d_cam), force_matrix_pred.shape[0], force_matrix_pred.shape[1])
    if obj_ids is None:
        obj_ids = list(range(n_objects))
    
    # 相机内参
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    
    # 遍历物体对，绘制力向量
    for i in range(n_objects):
        for j in range(n_objects):
            if i == j:  # 跳过自身
                continue
                
            force_pred = force_matrix_pred[i, j]
            force_gt = force_matrix_gt[i, j]
            
            force_pred_mag = np.linalg.norm(force_pred)
            force_gt_mag = np.linalg.norm(force_gt)
            
            # 只绘制超过阈值的力
            if force_pred_mag < min_force_magnitude and force_gt_mag < min_force_magnitude:
                continue
            
            # 获取物体i的位置作为力的起点
            obj_pos = poses_6d_cam[i][:3]  # [x, y, z]
            
            # 投影到图像坐标
            if obj_pos[2] > 0:  # 物体在相机前方
                x_img = int((obj_pos[0] / obj_pos[2]) * fx + cx)
                y_img = int((obj_pos[1] / obj_pos[2]) * fy + cy)
                
                if 0 <= x_img < img.shape[1] and 0 <= y_img < img.shape[0]:
                    # 计算力向量的终点
                    force_pred_end = obj_pos + force_pred * scale_factor
                    force_gt_end = obj_pos + force_gt * scale_factor
                    
                    # 投影力向量终点
                    if force_pred_mag >= min_force_magnitude and force_pred_end[2] > 0 and contact_matrix_pred[i, j] > 0.5:
                        x_pred_end = int((force_pred_end[0] / force_pred_end[2]) * fx + cx)
                        y_pred_end = int((force_pred_end[1] / force_pred_end[2]) * fy + cy)
                        
                        # 绘制预测力向量 (红色)
                        cv2.arrowedLine(img, (x_img, y_img), (x_pred_end, y_pred_end), 
                                      (0, 0, 255), 2, tipLength=0.3)
                        
                        # 添加标签
                        cv2.putText(img, f'P:{force_pred_mag:.3f}', (x_pred_end+5, y_pred_end-5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    
                    if force_gt_mag >= min_force_magnitude and force_gt_end[2] > 0:
                        x_gt_end = int((force_gt_end[0] / force_gt_end[2]) * fx + cx)
                        y_gt_end = int((force_gt_end[1] / force_gt_end[2]) * fy + cy)
                        
                        # 绘制真值力向量 (绿色)
                        cv2.arrowedLine(img, (x_img, y_img), (x_gt_end, y_gt_end), 
                                      (0, 255, 0), 2, tipLength=0.3)
                        
                        # 添加标签
                        cv2.putText(img, f'GT:{force_gt_mag:.3f}', (x_gt_end+5, y_gt_end+15),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    return img


def draw_environment_forces_on_image(rgb_image, env_force_pred, env_contact_pred, env_force_gt, 
                                   poses_6d_cam, camera_intrinsics, 
                                   scale_factor=0.02, min_force_magnitude=0.01, obj_ids=None):
    """
    在RGB图像上绘制环境力向量（物体对环境施加的力）
    
    Args:
        rgb_image: RGB图像 (H, W, 3)
        env_force_pred: 预测的环境力 (N, 3)
        env_contact_pred: 预测的环境接触概率 (N,)
        env_force_gt: 真值的环境力 (N, 3) 
        poses_6d_cam: 物体6D位姿 (N, 7) [x, y, z, qw, qx, qy, qz]
        camera_intrinsics: 相机内参矩阵 (3, 3)
        scale_factor: 力向量缩放因子
        min_force_magnitude: 最小力大小阈值
        obj_ids: 物体ID列表
        
    Returns:
        绘制了环境力向量的RGB图像
    """
    img = rgb_image.copy()
    
    if env_force_pred is None or env_force_gt is None or poses_6d_cam is None:
        return img
    
    n_objects = min(len(poses_6d_cam), env_force_pred.shape[0], env_force_gt.shape[0])
    if obj_ids is None:
        obj_ids = list(range(n_objects))
    
    # 相机内参
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    
    # 遍历物体，绘制环境力向量
    for i in range(n_objects):
        force_pred = env_force_pred[i]
        force_gt = env_force_gt[i]
        
        force_pred_mag = np.linalg.norm(force_pred)
        force_gt_mag = np.linalg.norm(force_gt)
        
        # 只绘制超过阈值的力
        if force_pred_mag < min_force_magnitude and force_gt_mag < min_force_magnitude:
            continue
        
        # 获取物体i的位置作为力的起点
        obj_pos = poses_6d_cam[i][:3]  # [x, y, z]
        
        # 投影到图像坐标
        if obj_pos[2] > 0:  # 物体在相机前方
            x_img = int((obj_pos[0] / obj_pos[2]) * fx + cx)
            y_img = int((obj_pos[1] / obj_pos[2]) * fy + cy)
            
            if 0 <= x_img < img.shape[1] and 0 <= y_img < img.shape[0]:
                # 绘制预测环境力向量 (蓝色)
                if force_pred_mag >= min_force_magnitude and (env_contact_pred is None or env_contact_pred[i] > 0.05):
                    force_pred_end = obj_pos + force_pred * scale_factor
                    if force_pred_end[2] > 0:
                        x_pred_end = int((force_pred_end[0] / force_pred_end[2]) * fx + cx)
                        y_pred_end = int((force_pred_end[1] / force_pred_end[2]) * fy + cy)
                        
                        # 绘制预测环境力向量 (蓝色)
                        cv2.arrowedLine(img, (x_img, y_img), (x_pred_end, y_pred_end), 
                                      (255, 0, 0), 3, tipLength=0.3)
                        
                        # 添加标签
                        cv2.putText(img, f'Env-P:{force_pred_mag:.3f}', (x_pred_end+5, y_pred_end-5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                
                # 绘制真值环境力向量 (青色)
                if force_gt_mag >= min_force_magnitude:
                    force_gt_end = obj_pos + force_gt * scale_factor
                    if force_gt_end[2] > 0:
                        x_gt_end = int((force_gt_end[0] / force_gt_end[2]) * fx + cx)
                        y_gt_end = int((force_gt_end[1] / force_gt_end[2]) * fy + cy)
                        
                        # 绘制真值环境力向量 (青色)
                        cv2.arrowedLine(img, (x_img, y_img), (x_gt_end, y_gt_end), 
                                      (255, 255, 0), 3, tipLength=0.3)
                        
                        # 添加标签
                        cv2.putText(img, f'Env-GT:{force_gt_mag:.3f}', (x_gt_end+5, y_gt_end+15),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    return img


def draw_6d_poses_comparison(rgb_image, poses_pred, poses_gt, camera_intrinsics, axis_length=0.05, obj_ids=None):
    """
    在RGB图像上绘制预测和真值的6D位姿
    
    Args:
        rgb_image: RGB图像 (H, W, 3)
        poses_pred: 预测的6D位姿 (N, 7) [x, y, z, qw, qx, qy, qz]
        poses_gt: 真值的6D位姿 (N, 7)
        camera_intrinsics: 相机内参矩阵 (3, 3)
        axis_length: 坐标轴长度
        obj_ids: 物体ID列表
        
    Returns:
        绘制了6D位姿的RGB图像
    """
    img = rgb_image.copy()
    
    if poses_pred is None or poses_gt is None:
        return img
    
    n_objects = len(poses_pred)
    if obj_ids is None:
        obj_ids = list(range(n_objects))
    
    # 相机内参
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    
    # 定义坐标轴向量和颜色
    axes = np.array([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]])
    pred_colors = [(0, 0, 255), (0, 255, 255), (255, 0, 255)]  # 预测：红、黄、紫 (BGR)
    gt_colors = [(0, 255, 0), (255, 255, 0), (255, 0, 0)]     # 真值：绿、青、蓝 (BGR)
    
    for i in range(n_objects):
        # 处理预测位姿
        pred_pos = poses_pred[i][:3]
        pred_quat = poses_pred[i][3:]  # [qw, qx, qy, qz]
        
        # 处理真值位姿
        gt_pos = poses_gt[i][:3] 
        gt_quat = poses_gt[i][3:]
        
        # 转换四元数为旋转矩阵
        if np.linalg.norm(pred_quat) > 0:
            pred_quat_normalized = pred_quat / np.linalg.norm(pred_quat)
            # scipy expects [x, y, z, w]
            pred_quat_xyzw = [pred_quat_normalized[1], pred_quat_normalized[2], 
                             pred_quat_normalized[3], pred_quat_normalized[0]]
            pred_R = R.from_quat(pred_quat_xyzw).as_matrix()
        else:
            pred_R = np.eye(3)
            
        if np.linalg.norm(gt_quat) > 0:
            gt_quat_normalized = gt_quat / np.linalg.norm(gt_quat)
            gt_quat_xyzw = [gt_quat_normalized[1], gt_quat_normalized[2], 
                           gt_quat_normalized[3], gt_quat_normalized[0]]
            gt_R = R.from_quat(gt_quat_xyzw).as_matrix()
        else:
            gt_R = np.eye(3)
        
        # 绘制预测位姿的坐标轴
        if pred_pos[2] > 0:
            pred_center_2d = project_point_to_image(pred_pos, fx, fy, cx, cy)
            if pred_center_2d is not None:
                # 绘制预测坐标轴
                for axis_idx, axis in enumerate(axes):
                    axis_world = pred_R @ axis + pred_pos
                    if axis_world[2] > 0:
                        axis_2d = project_point_to_image(axis_world, fx, fy, cx, cy)
                        if axis_2d is not None:
                            cv2.arrowedLine(img, tuple(pred_center_2d), tuple(axis_2d), 
                                          pred_colors[axis_idx], 2, tipLength=0.3)
                
                # 绘制物体中心点 (预测)
                cv2.circle(img, tuple(pred_center_2d), 4, (0, 0, 255), -1)
                cv2.putText(img, f'P{obj_ids[i]}', (pred_center_2d[0]+8, pred_center_2d[1]-8),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 绘制真值位姿的坐标轴
        if gt_pos[2] > 0:
            gt_center_2d = project_point_to_image(gt_pos, fx, fy, cx, cy)
            if gt_center_2d is not None:
                # 绘制真值坐标轴
                for axis_idx, axis in enumerate(axes):
                    axis_world = gt_R @ axis + gt_pos
                    if axis_world[2] > 0:
                        axis_2d = project_point_to_image(axis_world, fx, fy, cx, cy)
                        if axis_2d is not None:
                            cv2.arrowedLine(img, tuple(gt_center_2d), tuple(axis_2d), 
                                          gt_colors[axis_idx], 2, tipLength=0.3)
                
                # 绘制物体中心点 (真值)
                cv2.circle(img, tuple(gt_center_2d), 4, (0, 255, 0), -1)
                cv2.putText(img, f'GT{obj_ids[i]}', (gt_center_2d[0]+8, gt_center_2d[1]+8),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return img


def project_point_to_image(point_3d, fx, fy, cx, cy):
    """将3D点投影到2D图像坐标"""
    if point_3d[2] <= 0:
        return None
        
    x = int((point_3d[0] / point_3d[2]) * fx + cx)
    y = int((point_3d[1] / point_3d[2]) * fy + cy)
    
    # 检查是否在图像范围内
    if x < 0 or y < 0:
        return None
        
    return np.array([x, y], dtype=int)


def save_visualization(image, output_path, sample_id, batch_idx, img_idx, viz_type):
    """保存可视化图像"""
    filename = f"sample_{sample_id:06d}_batch_{batch_idx:03d}_img_{img_idx:03d}_{viz_type}.png"
    full_path = os.path.join(output_path, filename)
    
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    
    # 保存图像
    cv2.imwrite(full_path, image)
    return full_path


def _reorder_eval_matrix(pred_matrix_item, n_gt_objects, src_idx, tgt_idx, n_obj_queries):
    """
    Reorders a prediction matrix from query space to ground truth space for evaluation, including the environment.
    This function is similar to the _reorder_matrix function in pose_estimation_transformer.py but adapted for evaluation.
    
    Args:
        pred_matrix_item: Single prediction matrix from the batch [n_obj_queries+1, n_obj_queries+1, C]
        n_gt_objects: Number of ground truth objects in the current sample.
        src_idx, tgt_idx: Matching indices from Hungarian matcher for objects.
        n_obj_queries: Number of object queries (excluding environment query).
    
    Returns:
        reordered_matrix: Reordered matrix of size [n_gt_objects+1, n_gt_objects+1, C].
    """
    device = pred_matrix_item.device if hasattr(pred_matrix_item, 'device') else 'cpu'
    C = pred_matrix_item.shape[-1]
    n_total_gt_entities = n_gt_objects + 1  # +1 for environment
    
    # Convert to numpy if it's a tensor
    if hasattr(pred_matrix_item, 'cpu'):
        pred_matrix_np = pred_matrix_item.detach().cpu().numpy()
    else:
        pred_matrix_np = pred_matrix_item
    
    reordered_matrix = np.zeros((n_total_gt_entities, n_total_gt_entities, C), dtype=np.float32)

    # The environment query is at index `n_obj_queries`
    env_query_idx = n_obj_queries

    # Map matched object-object predictions
    for t1, s1 in zip(tgt_idx, src_idx):
        for t2, s2 in zip(tgt_idx, src_idx):
            if s1 < pred_matrix_np.shape[0] and s2 < pred_matrix_np.shape[1]:
                reordered_matrix[t1, t2] = pred_matrix_np[s1, s2]
    
    # Map object-environment interactions
    for t_obj, s_obj in zip(tgt_idx, src_idx):
        if s_obj < pred_matrix_np.shape[0] and env_query_idx < pred_matrix_np.shape[1]:
            # Object to environment
            reordered_matrix[t_obj, n_gt_objects] = pred_matrix_np[s_obj, env_query_idx]
        if env_query_idx < pred_matrix_np.shape[0] and s_obj < pred_matrix_np.shape[1]:
            # Environment to object
            reordered_matrix[n_gt_objects, t_obj] = pred_matrix_np[env_query_idx, s_obj]

    # Map environment-environment self-interaction
    if env_query_idx < pred_matrix_np.shape[0] and env_query_idx < pred_matrix_np.shape[1]:
        reordered_matrix[n_gt_objects, n_gt_objects] = pred_matrix_np[env_query_idx, env_query_idx]

    return reordered_matrix


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
    
    # 可视化设置
    enable_viz = getattr(args, 'enable_visualization', True)
    viz_sample_rate = getattr(args, 'viz_sample_rate', 0.01)
    viz_output_dir = getattr(args, 'viz_output_dir', './results/visualizations')
    viz_counter = 0
    
    if enable_viz:
        os.makedirs(viz_output_dir, exist_ok=True)
        print(f"Visualization enabled with {viz_sample_rate*100:.1f}% sample rate")
        print(f"Visualization output directory: {viz_output_dir}")
    
    print("Starting to collect prediction data...")
    print(f"Dataset has {len(data_loader)} batches in total")
    
    # 收集接触分类数据
    all_contact_probs = []
    all_contact_labels = []
    
    # 收集力回归数据（仅在有接触的点上）
    all_force_predictions = []  # 预测的力向量
    all_force_ground_truth = []  # 真值力向量
    all_contact_mask = []  # 接触掩码，用于条件回归评估
    
    # 收集环境力数据（物体与环境之间的接触和力）
    all_env_contact_probs = []  # 环境接触预测概率 [N] float32
    all_env_contact_labels = []  # 环境接触真值标签 [N] bool
    all_env_force_predictions = []  # 环境力预测 [N, 3] float32
    all_env_force_ground_truth = []  # 环境力真值 [N, 3] float32
    all_env_contact_mask = []  # 环境接触掩码 [N] bool
    
    for batch_idx, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        outputs, n_boxes_per_sample = model(samples, targets)
        
        if outputs is None:
            continue
            
        if not any(t['labels'].numel() > 0 for t in targets):
            continue
            
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        
        # 获取matcher的indices
        indices = matcher(outputs_without_aux, targets, n_boxes_per_sample)
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
                    # print(f"target_matrix shape: {target_matrix.shape}, ")
                    if n_matched > 0:
                        # 参考pose_estimation_transformer.py中的_reorder_matrix处理逻辑
                        n_objects = target_matrix.shape[0] - 1  # Ground truth size
                        
                        # 初始化重排序矩阵变量
                        reordered_force_matrix = None
                        reordered_contact_logits = None
                        
                        # 如果预测矩阵包含环境节点（N+1维度），则使用完整的重排序
                        if batch_pred_force_matrix.shape[0] > n_objects:
                            # 使用_reorder_eval_matrix函数进行完整重排序，包括环境交互
                            reordered_force_matrix = _reorder_eval_matrix(
                                batch_pred_force_matrix, n_objects, src_idx, tgt_idx, args.num_queries
                            )
                            reordered_contact_logits = _reorder_eval_matrix(
                                batch_pred_contact_logits, n_objects, src_idx, tgt_idx, args.num_queries
                            )
                            
                            # 提取对象-对象交互部分（排除环境）
                            matched_pred_force_matrix = reordered_force_matrix[:n_objects, :n_objects, :]
                            matched_pred_contact_logits = reordered_contact_logits[:n_objects, :n_objects, :]
                        else:
                            # 如果没有环境节点，使用传统的重排序方法
                            reordered_pred_force_matrix = np.zeros((n_objects, n_objects, 3), dtype=np.float32)
                            reordered_pred_contact_logits = np.zeros((n_objects, n_objects, 1), dtype=np.float32)
                            
                            # 使用indices映射预测到ground truth位置
                            for idx_i, tgt_i in enumerate(tgt_idx):
                                for idx_j, tgt_j in enumerate(tgt_idx):
                                    if idx_i < len(src_idx) and idx_j < len(src_idx):
                                        src_i = src_idx[idx_i]
                                        src_j = src_idx[idx_j]
                                        if src_i < batch_pred_force_matrix.shape[0] and src_j < batch_pred_force_matrix.shape[1]:
                                            reordered_pred_force_matrix[tgt_i, tgt_j] = batch_pred_force_matrix[src_i, src_j]
                                            # 处理contact logits的维度
                                            if batch_pred_contact_logits.ndim == 3:
                                                reordered_pred_contact_logits[tgt_i, tgt_j] = batch_pred_contact_logits[src_i, src_j]
                                            else:
                                                reordered_pred_contact_logits[tgt_i, tgt_j, 0] = batch_pred_contact_logits[src_i, src_j]
                            
                            matched_pred_force_matrix = reordered_pred_force_matrix
                            matched_pred_contact_logits = reordered_pred_contact_logits
                        matched_target_matrix = target_matrix[:n_objects, :n_objects]  # 只取对象-对象部分
                        
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
                            print(f"Ground truth contact matrix: {gt_contact_matrix}")
                            print(f"Predicted contact probs: {pred_contact_probs}")
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
                        
                        # 确保数据维度正确：NxN -> N*N，并排除对角线元素
                        n_obj = gt_contact_matrix.shape[0]
                        non_diagonal_mask = ~np.eye(n_obj, dtype=bool)

                        pred_contact_flat = pred_contact_probs[non_diagonal_mask]  # [N*(N-1)]
                        gt_contact_flat = gt_contact_matrix[non_diagonal_mask]  # [N*(N-1)] bool类型
                        
                        all_contact_probs.extend(pred_contact_flat)
                        all_contact_labels.extend(gt_contact_flat)
                        
                        # === 可视化处理 ===
                        if enable_viz and random.random() < viz_sample_rate:
                            # 获取RGB图像和相机信息
                            if hasattr(samples, 'tensors'):
                                rgb_tensor = samples.tensors[i]  # [3, H, W]
                                rgb_image = rgb_tensor.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
                                rgb_image = (rgb_image * 255).astype(np.uint8)
                                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                                # 获取相机内参 (假设在targets中)
                                if 'camera_intrinsic' in t:
                                    camera_intrinsics = t['camera_intrinsic'].cpu().numpy()
                                else:
                                    # 使用默认相机内参
                                    h, w = rgb_image.shape[:2]
                                    fx = fy = max(h, w) * 0.7  # 估计焦距
                                    cx, cy = w / 2, h / 2
                                    camera_intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                                
                                # 获取6D位姿预测和真值
                                if 'pred_translation' in outputs_without_aux and 'pred_rotation' in outputs_without_aux:
                                    pred_translations = outputs_without_aux['pred_translation'][i].cpu().numpy()  # [N_queries, 3]
                                    pred_rotations = outputs_without_aux['pred_rotation'][i].cpu().numpy()      # [N_queries, 3, 3] or [N_queries, 4]
                                    
                                    # 构建预测6D位姿 [N, 7] 格式
                                    poses_pred = []
                                    for src_idx_item in src_idx:
                                        if src_idx_item < len(pred_translations):
                                            trans = pred_translations[src_idx_item]
                                            rot = pred_rotations[src_idx_item]
                                            
                                            # 转换旋转矩阵到四元数
                                            if rot.shape == (3, 3):
                                                # 旋转矩阵到四元数 [w, x, y, z]
                                                from scipy.spatial.transform import Rotation as R_scipy
                                                quat_wxyz = R_scipy.from_matrix(rot).as_quat()  # [x,y,z,w]
                                                quat = np.array([quat_wxyz[3], quat_wxyz[0], quat_wxyz[1], quat_wxyz[2]])  # [w,x,y,z]
                                            else:
                                                quat = rot[:4]  # 假设已经是四元数格式
                                            
                                            pose = np.concatenate([trans, quat])  # [x,y,z,qw,qx,qy,qz]
                                            poses_pred.append(pose)
                                    
                                    poses_pred = np.array(poses_pred) if poses_pred else np.zeros((0, 7))
                                    
                                    # 获取真值6D位姿
                                    if 'relative_position' in t and 'relative_rotation' in t:
                                        gt_translations = t['relative_position'].cpu().numpy()  # [N_gt, 3]
                                        gt_rotations = t['relative_rotation'].cpu().numpy()      # [N_gt, 3, 3]
                                        
                                        poses_gt = []
                                        for gt_idx in range(len(gt_translations)):
                                            trans = gt_translations[gt_idx]
                                            rot_matrix = gt_rotations[gt_idx]
                                            
                                            # 转换旋转矩阵到四元数
                                            from scipy.spatial.transform import Rotation as R_scipy
                                            quat_wxyz = R_scipy.from_matrix(rot_matrix).as_quat()  # [x,y,z,w]
                                            quat = np.array([quat_wxyz[3], quat_wxyz[0], quat_wxyz[1], quat_wxyz[2]])  # [w,x,y,z]
                                            
                                            pose = np.concatenate([trans, quat])
                                            poses_gt.append(pose)
                                        
                                        poses_gt = np.array(poses_gt) if poses_gt else np.zeros((0, 7))
                                    else:
                                        poses_gt = np.zeros((0, 7))
                                    
                                    # 绘制力向量
                                    force_viz_image = draw_forces_on_image(
                                        rgb_image,
                                        matched_pred_force_matrix,  # 预测力矩阵
                                        matched_pred_contact_logits,  # 预测接触logits
                                        matched_target_matrix,      # 真值力矩阵
                                        poses_gt if len(poses_gt) > 0 else poses_pred,  # 使用真值位姿，如果没有则用预测
                                        camera_intrinsics,
                                        scale_factor=getattr(args, 'viz_force_scale', 0.02),
                                        min_force_magnitude=getattr(args, 'viz_min_force', 0.01)
                                    )
                                    
                                    # 绘制环境力向量（如果有环境力数据）
                                    env_force_viz_image = rgb_image.copy()
                                    if ('environment_forces' in t and t['environment_forces'] is not None and 
                                        reordered_force_matrix is not None and 
                                        reordered_contact_logits is not None and
                                        reordered_force_matrix.shape[0] > n_objects):
                                        
                                        env_target_matrix = target_matrix[:n_objects, n_objects, :]  # 只取环境力部分
                                        pred_env_force_matrix = reordered_force_matrix[:n_objects, n_objects, :].astype(np.float32)  # [N, 3]
                                        pred_env_contact_logits = reordered_contact_logits[:n_objects, n_objects]  # [N] or [N, 1]
                                        
                                        # 处理环境接触logits的维度
                                        if pred_env_contact_logits.ndim == 2 and pred_env_contact_logits.shape[-1] == 1:
                                            pred_env_contact_logits = pred_env_contact_logits.squeeze(-1)
                                        
                                        # 转换环境接触预测为概率
                                        pred_env_contact_probs = torch.sigmoid(torch.tensor(pred_env_contact_logits)).numpy()
                                        
                                        env_force_viz_image = draw_environment_forces_on_image(
                                            rgb_image,
                                            pred_env_force_matrix,  # 预测环境力
                                            pred_env_contact_probs,  # 预测环境接触概率
                                            env_target_matrix,      # 真值环境力
                                            poses_gt if len(poses_gt) > 0 else poses_pred,  # 使用真值位姿，如果没有则用预测
                                            camera_intrinsics,
                                            scale_factor=getattr(args, 'viz_force_scale', 0.02),
                                            min_force_magnitude=getattr(args, 'viz_min_force', 0.01)
                                        )
                                    
                                    # 绘制6D位姿对比
                                    pose_viz_image = draw_6d_poses_comparison(
                                        rgb_image,
                                        poses_pred,
                                        poses_gt, 
                                        camera_intrinsics,
                                        axis_length=0.05
                                    )
                                    
                                    # 保存可视化结果
                                    save_visualization(force_viz_image, viz_output_dir, viz_counter, batch_idx, i, "force_comparison")
                                    save_visualization(env_force_viz_image, viz_output_dir, viz_counter, batch_idx, i, "env_force_comparison")
                                    save_visualization(pose_viz_image, viz_output_dir, viz_counter, batch_idx, i, "pose_comparison")
                                    
                                    if viz_counter % 10 == 0:
                                        print(f"Saved visualization {viz_counter}: batch {batch_idx}, sample {i}")
                                    
                                    viz_counter += 1
                        
                        # 收集力回归数据 - 同样排除对角线
                        pred_force_flat = matched_pred_force_matrix[non_diagonal_mask].reshape(-1, 3).astype(np.float32)
                        gt_force_flat = matched_target_matrix[non_diagonal_mask].reshape(-1, 3).astype(np.float32)
                        contact_mask_flat = gt_contact_flat # 接触掩码，已经排除了对角线
                        
                        all_force_predictions.extend(pred_force_flat)
                        all_force_ground_truth.extend(gt_force_flat)
                        all_contact_mask.extend(contact_mask_flat)
                        
                        # === 新增：环境力和环境接触处理 ===
                        # 使用重排序后的完整矩阵来处理环境交互
                        if (batch_pred_force_matrix.shape[0] > n_objects and 
                            reordered_force_matrix is not None and
                            reordered_contact_logits is not None and
                            'environment_forces' in t and 
                            t['environment_forces'] is not None):
                            
                            env_target_matrix = t['environment_forces'].detach().cpu().numpy()  # [N, 3]
                            # print(f"Environment target matrix {env_target_matrix}, env_target_matrix: {target_matrix[:n_objects, n_objects, :]}")
                            # 从重排序后的完整矩阵中提取环境交互
                            # 提取物体到环境的力 (最后一列，前n_objects行)
                            pred_env_force_matrix = reordered_force_matrix[:n_objects, n_objects, :].astype(np.float32)  # [N, 3]
                            pred_env_contact_logits = reordered_contact_logits[:n_objects, n_objects]  # [N] or [N, 1]
                            
                            # 处理环境接触logits的维度
                            if pred_env_contact_logits.ndim == 2 and pred_env_contact_logits.shape[-1] == 1:
                                pred_env_contact_logits = pred_env_contact_logits.squeeze(-1)
                            
                            # 转换环境接触预测为概率
                            pred_env_contact_probs = torch.sigmoid(torch.tensor(pred_env_contact_logits)).numpy()
                            
                            # 创建环境接触标签（基于环境力的大小）
                            gt_env_force_magnitude = np.linalg.norm(env_target_matrix, axis=-1)
                            gt_env_contact_labels = (gt_env_force_magnitude > contact_threshold).astype(bool)
                            
                            # Debug: Check lengths before collecting
                            if i == 0 and batch_idx < 3:
                                print(f"DEBUG: pred_env_contact_probs shape: {pred_env_contact_probs.shape}")
                                print(f"DEBUG: gt_env_contact_labels shape: {gt_env_contact_labels.shape}")
                                print(f"DEBUG: Before extend - all_env_contact_probs len: {len(all_env_contact_probs)}")
                                print(f"DEBUG: Before extend - all_env_contact_labels len: {len(all_env_contact_labels)}")
                            
                            # 收集环境力和环境接触数据
                            all_env_contact_probs.extend(pred_env_contact_probs)
                            all_env_contact_labels.extend(gt_env_contact_labels)
                            all_env_force_predictions.extend(pred_env_force_matrix)
                            all_env_force_ground_truth.extend(env_target_matrix)
                            all_env_contact_mask.extend(gt_env_contact_labels)
                            
                            # Debug: Check lengths after collecting
                            if i == 0 and batch_idx < 3:
                                print(f"DEBUG: After extend - all_env_contact_probs len: {len(all_env_contact_probs)}")
                                print(f"DEBUG: After extend - all_env_contact_labels len: {len(all_env_contact_labels)}")
                            
                            # 调试信息（仅在前几个batch）
                            if i == 0 and batch_idx < 3:
                                print(f"\n=== ENVIRONMENT DEBUG INFO for batch {batch_idx}, sample {i} ===")
                                print(f"Environment target matrix shape: {env_target_matrix.shape}")
                                print(f"Predicted env contact logits shape: {pred_env_contact_logits.shape}")
                                print(f"Environment contact labels: {gt_env_contact_labels}")
                                print(f"Environment contact probabilities: {pred_env_contact_probs}")
                                print(f"Environment force magnitude stats: min={gt_env_force_magnitude.min():.6f}, max={gt_env_force_magnitude.max():.6f}")
                                print(f"Environment contact ratio: {gt_env_contact_labels.mean():.4f} ({np.sum(gt_env_contact_labels)}/{len(gt_env_contact_labels)})")
                        
                        # === 环境力处理结束 ===
                        
                        if i == 0 and batch_idx % 20 == 0:
                            print(f"Matched shapes - contact: {matched_pred_contact_logits.shape}, force: {matched_pred_force_matrix.shape}")
                            print(f"Contact positive ratio: {gt_contact_matrix.mean():.4f}")
                            print(f"Force magnitude range: [{gt_force_magnitude.min():.6f}, {gt_force_magnitude.max():.6f}]")
    
    print(f"Collected {len(all_contact_probs)} contact prediction samples")
    print(f"Collected {len(all_force_predictions)} force prediction samples")
    print(f"Collected {len(all_env_contact_probs)} environment contact prediction samples")
    print(f"Collected {len(all_env_force_predictions)} environment force prediction samples")
    
    # Debug: Check environment data lengths before conversion
    print(f"DEBUG: Raw all_env_contact_probs length: {len(all_env_contact_probs)}")
    print(f"DEBUG: Raw all_env_contact_labels length: {len(all_env_contact_labels)}")
    
    # Check for potential duplication or inconsistency in raw data
    if len(all_env_contact_probs) != len(all_env_contact_labels):
        print(f"WARNING: Environment data length mismatch detected during collection!")
        print(f"This suggests duplicate data collection or incomplete data extraction.")
    
    # 转换为numpy数组，确保正确的数据类型
    all_contact_probs = np.array(all_contact_probs, dtype=np.float32)  # [N*N] float32
    all_contact_labels = np.array(all_contact_labels, dtype=bool)      # [N*N] bool
    all_force_predictions = np.array(all_force_predictions, dtype=np.float32)  # [N*N, 3] float32
    all_force_ground_truth = np.array(all_force_ground_truth, dtype=np.float32) # [N*N, 3] float32
    all_contact_mask = np.array(all_contact_mask, dtype=bool)          # [N*N] bool
    
    # 转换环境力和环境接触数据
    if len(all_env_contact_probs) > 0:
        # Final safety check before numpy conversion
        if len(all_env_contact_probs) != len(all_env_contact_labels):
            print(f"FINAL WARNING: Fixing length mismatch before numpy conversion")
            print(f"all_env_contact_probs: {len(all_env_contact_probs)}, all_env_contact_labels: {len(all_env_contact_labels)}")
            
            min_length = min(len(all_env_contact_probs), len(all_env_contact_labels))
            all_env_contact_probs = all_env_contact_probs[:min_length]
            all_env_contact_labels = all_env_contact_labels[:min_length]
            all_env_force_predictions = all_env_force_predictions[:min_length]
            all_env_force_ground_truth = all_env_force_ground_truth[:min_length]
            all_env_contact_mask = all_env_contact_mask[:min_length]
            
            print(f"All environment arrays trimmed to length: {min_length}")
        
        all_env_contact_probs = np.array(all_env_contact_probs, dtype=np.float32)  # [N] float32
        all_env_contact_labels = np.array(all_env_contact_labels, dtype=bool)      # [N] bool
        all_env_force_predictions = np.array(all_env_force_predictions, dtype=np.float32)  # [N, 3] float32
        all_env_force_ground_truth = np.array(all_env_force_ground_truth, dtype=np.float32) # [N, 3] float32
        all_env_contact_mask = np.array(all_env_contact_mask, dtype=bool)          # [N] bool
    else:
        # 创建空数组以保持接口一致
        all_env_contact_probs = np.array([], dtype=np.float32)
        all_env_contact_labels = np.array([], dtype=bool)
        all_env_force_predictions = np.zeros((0, 3), dtype=np.float32)
        all_env_force_ground_truth = np.zeros((0, 3), dtype=np.float32)
        all_env_contact_mask = np.array([], dtype=bool)
    
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
    
    # 环境力和环境接触数据分布信息
    if len(all_env_contact_probs) > 0:
        print(f"\n=== Environment data distribution debugging info ===")
        print(f"Environment contact prediction probability range: [{all_env_contact_probs.min():.6f}, {all_env_contact_probs.max():.6f}]")
        print(f"Environment contact positive samples: {np.sum(all_env_contact_labels)} / {len(all_env_contact_labels)} ({np.mean(all_env_contact_labels.astype(float)):.4f})")
        print(f"Environment force prediction range: [{all_env_force_predictions.min():.6f}, {all_env_force_predictions.max():.6f}]")
        print(f"Environment force ground truth range: [{all_env_force_ground_truth.min():.6f}, {all_env_force_ground_truth.max():.6f}]")
        print(f"Environment contact points for force evaluation: {np.sum(all_env_contact_mask)}")
        print(f"Environment data shapes - contact_probs: {all_env_contact_probs.shape}, contact_labels: {all_env_contact_labels.shape}")
        print(f"Environment data shapes - force_pred: {all_env_force_predictions.shape}, force_gt: {all_env_force_ground_truth.shape}")
    else:
        print(f"\n=== Environment data ===")
        print(f"No environment force/contact data found in the dataset")
    
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
        'contact_mask': all_contact_mask,
        # 环境力和环境接触数据
        'env_contact_probs': all_env_contact_probs,
        'env_contact_labels': all_env_contact_labels,
        'env_force_predictions': all_env_force_predictions,
        'env_force_ground_truth': all_env_force_ground_truth,
        'env_contact_mask': all_env_contact_mask
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


def evaluate_environment_contact_classification(env_contact_probs, env_contact_labels, threshold=0.5):
    """
    评估环境接触分类性能
    
    Args:
        env_contact_probs: 预测的环境接触概率 [N] float32
        env_contact_labels: 真值环境接触标签 [N] bool
        threshold: 分类阈值
    
    Returns:
        dict: 包含环境接触分类指标的字典
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
    
    # Debug: Check array lengths before processing
    print(f"DEBUG: env_contact_probs length: {len(env_contact_probs)}")
    print(f"DEBUG: env_contact_labels length: {len(env_contact_labels)}")
    
    if len(env_contact_probs) == 0:
        return {
            'threshold': threshold,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'accuracy': 0.0,
            'true_positives': 0,
            'true_negatives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'total_samples': 0,
            'positive_samples': 0,
            'negative_samples': 0
        }
    
    # Check for length mismatch and fix it
    if len(env_contact_probs) != len(env_contact_labels):
        print(f"WARNING: Length mismatch detected!")
        print(f"env_contact_probs: {len(env_contact_probs)}, env_contact_labels: {len(env_contact_labels)}")
        
        # Take the minimum length to avoid index errors
        min_length = min(len(env_contact_probs), len(env_contact_labels))
        env_contact_probs = env_contact_probs[:min_length]
        env_contact_labels = env_contact_labels[:min_length]
        
        print(f"Trimmed both arrays to length: {min_length}")
    
    if len(env_contact_probs) == 0:
        return {
            'threshold': threshold,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'accuracy': 0.0,
            'true_positives': 0,
            'true_negatives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'total_samples': 0,
            'positive_samples': 0,
            'negative_samples': 0
        }
    
    # 转换概率为二分类预测
    predictions = (env_contact_probs >= threshold).astype(bool)
    
    # 转换bool标签为int用于sklearn计算
    env_contact_labels_int = env_contact_labels.astype(int)
    predictions_int = predictions.astype(int)
    
    # 计算分类指标
    precision = precision_score(env_contact_labels_int, predictions_int, zero_division=0)
    recall = recall_score(env_contact_labels_int, predictions_int, zero_division=0)
    f1 = f1_score(env_contact_labels_int, predictions_int, zero_division=0)
    accuracy = accuracy_score(env_contact_labels_int, predictions_int)
    
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(env_contact_labels_int, predictions_int).ravel()
    
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
        'total_samples': len(env_contact_labels),
        'positive_samples': int(np.sum(env_contact_labels)),
        'negative_samples': int(np.sum(~env_contact_labels))
    }
    
    print(f"\n=== Environment Contact Classification Results (threshold={threshold:.3f}) ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"True Positives: {tp}, True Negatives: {tn}")
    print(f"False Positives: {fp}, False Negatives: {fn}")
    print(f"Total samples: {len(env_contact_labels)}, Positive: {np.sum(env_contact_labels)}, Negative: {np.sum(~env_contact_labels)}")
    
    # 分析类别不平衡问题
    positive_ratio = np.sum(env_contact_labels) / len(env_contact_labels)
    print(f"\n=== Class Imbalance Analysis ===")
    print(f"Positive class ratio: {positive_ratio:.4f} ({positive_ratio*100:.1f}%)")
    print(f"Negative class ratio: {1-positive_ratio:.4f} ({(1-positive_ratio)*100:.1f}%)")
    
    if positive_ratio > 0.8:
        print(f"WARNING: Severe class imbalance detected! {positive_ratio*100:.1f}% positive samples")
        print(f"Consider using a lower threshold (e.g., 0.1-0.3) for better balance")
        print(f"Current threshold {threshold:.3f} may be too conservative")
    elif positive_ratio < 0.2:
        print(f"WARNING: Severe class imbalance detected! {(1-positive_ratio)*100:.1f}% negative samples")
        print(f"Consider using a higher threshold (e.g., 0.7-0.9) for better balance")
        print(f"Current threshold {threshold:.3f} may be too permissive")
    
    # 计算平衡精确度（考虑类别不平衡）
    balanced_accuracy = (recall + tn/(tn+fp)) / 2 if (tn+fp) > 0 else 0.0
    print(f"Balanced Accuracy: {balanced_accuracy:.4f} (considers class imbalance)")
    
    return results


def analyze_optimal_threshold_for_imbalanced_data(contact_probs, contact_labels, metric='f1'):
    """
    分析类别不平衡数据的最优阈值
    
    Args:
        contact_probs: 预测概率 [N] float32
        contact_labels: 真值标签 [N] bool
        metric: 优化的指标 ('f1', 'balanced_accuracy', 'precision', 'recall')
    
    Returns:
        dict: 最优阈值分析结果
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    
    print(f"\n=== Optimal Threshold Analysis for Imbalanced Data ===")
    
    # Debug: Check array lengths before processing
    print(f"DEBUG: contact_probs length: {len(contact_probs)}")
    print(f"DEBUG: contact_labels length: {len(contact_labels)}")
    
    # Check for length mismatch and fix it
    if len(contact_probs) != len(contact_labels):
        print(f"WARNING: Length mismatch detected in threshold analysis!")
        print(f"contact_probs: {len(contact_probs)}, contact_labels: {len(contact_labels)}")
        
        # Take the minimum length to avoid index errors
        min_length = min(len(contact_probs), len(contact_labels))
        contact_probs = contact_probs[:min_length]
        contact_labels = contact_labels[:min_length]
        
        print(f"Trimmed both arrays to length: {min_length}")
    
    if len(contact_probs) == 0:
        return {
            'best_threshold': 0.5,
            'best_score': 0.0,
            'metric': metric,
            'analysis_results': {'thresholds': [], 'precisions': [], 'recalls': [], 'f1_scores': [], 'accuracies': [], 'balanced_accuracies': []},
            'positive_ratio': 0.0
        }
    
    # 测试不同的阈值
    test_thresholds = np.arange(0.05, 0.95, 0.05)
    
    best_threshold = 0.5
    best_score = 0.0
    
    results = {
        'thresholds': [],
        'precisions': [],
        'recalls': [],
        'f1_scores': [],
        'accuracies': [],
        'balanced_accuracies': []
    }
    
    print(f"{'Threshold':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Accuracy':<10} {'Balanced_Acc':<12}")
    print("-" * 72)
    
    for threshold in test_thresholds:
        predictions = (contact_probs >= threshold).astype(bool)
        labels_int = contact_labels.astype(int)
        predictions_int = predictions.astype(int)
        
        precision = precision_score(labels_int, predictions_int, zero_division=0)
        recall = recall_score(labels_int, predictions_int, zero_division=0)
        f1 = f1_score(labels_int, predictions_int, zero_division=0)
        accuracy = accuracy_score(labels_int, predictions_int)
        
        # 计算平衡精确度
        tn = np.sum((predictions_int == 0) & (labels_int == 0))
        fp = np.sum((predictions_int == 1) & (labels_int == 0))
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        balanced_accuracy = (recall + specificity) / 2
        
        results['thresholds'].append(threshold)
        results['precisions'].append(precision)
        results['recalls'].append(recall)
        results['f1_scores'].append(f1)
        results['accuracies'].append(accuracy)
        results['balanced_accuracies'].append(balanced_accuracy)
        
        print(f"{threshold:<10.2f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {accuracy:<10.4f} {balanced_accuracy:<12.4f}")
        
        # 根据指定指标选择最佳阈值
        if metric == 'f1' and f1 > best_score:
            best_score = f1
            best_threshold = threshold
        elif metric == 'balanced_accuracy' and balanced_accuracy > best_score:
            best_score = balanced_accuracy
            best_threshold = threshold
        elif metric == 'precision' and precision > best_score:
            best_score = precision
            best_threshold = threshold
        elif metric == 'recall' and recall > best_score:
            best_score = recall
            best_threshold = threshold
    
    print(f"\nBest threshold: {best_threshold:.2f} (optimizing {metric}: {best_score:.4f})")
    
    # 给出建议
    positive_ratio = np.mean(contact_labels.astype(float))
    print(f"\n=== Recommendations ===")
    print(f"Dataset positive ratio: {positive_ratio:.4f}")
    
    if positive_ratio > 0.8:
        print("- Dataset is heavily imbalanced towards positive class")
        print("- Consider using a lower threshold (0.1-0.3) to improve recall")
        print("- F1-score or balanced accuracy may be better metrics than accuracy")
    elif positive_ratio < 0.2:
        print("- Dataset is heavily imbalanced towards negative class") 
        print("- Consider using a higher threshold (0.7-0.9) to improve precision")
        print("- F1-score or balanced accuracy may be better metrics than accuracy")
    else:
        print("- Dataset has reasonable balance")
        print("- Standard threshold of 0.5 should work well")
    
    return {
        'best_threshold': best_threshold,
        'best_score': best_score,
        'metric': metric,
        'analysis_results': results,
        'positive_ratio': positive_ratio
    }


def evaluate_environment_force_regression(env_force_predictions, env_force_ground_truth, env_contact_mask):
    """
    评估环境力回归性能（仅在有环境接触的物体上）
    
    Args:
        env_force_predictions: 预测的环境力向量 [N, 3] float32
        env_force_ground_truth: 真值环境力向量 [N, 3] float32
        env_contact_mask: 环境接触掩码 [N] bool，True表示有环境接触，False表示无环境接触
    
    Returns:
        dict: 包含环境力回归指标的字典
    """
    if len(env_force_predictions) == 0:
        return {
            'mae': 0.0,
            'rmse': 0.0,
            'mae_per_axis': [0.0, 0.0, 0.0],
            'rmse_per_axis': [0.0, 0.0, 0.0],
            'magnitude_mae': 0.0,
            'magnitude_rmse': 0.0,
            'mean_cosine_similarity': 0.0,
            'mean_angular_error_deg': 0.0,
            'contact_points': 0,
            'total_points': 0,
            'valid_direction_points': 0
        }
    
    # 仅选择有环境接触的物体
    contact_indices = env_contact_mask == True
    
    if np.sum(contact_indices) == 0:
        return {
            'mae': 0.0,
            'rmse': 0.0,
            'mae_per_axis': [0.0, 0.0, 0.0],
            'rmse_per_axis': [0.0, 0.0, 0.0],
            'magnitude_mae': 0.0,
            'magnitude_rmse': 0.0,
            'mean_cosine_similarity': 0.0,
            'mean_angular_error_deg': 0.0,
            'contact_points': 0,
            'total_points': len(env_contact_mask),
            'valid_direction_points': 0
        }
    
    contact_pred_forces = env_force_predictions[contact_indices]
    contact_gt_forces = env_force_ground_truth[contact_indices]
    
    # 计算整体误差指标
    mae = np.mean(np.abs(contact_pred_forces - contact_gt_forces))
    mse = np.mean((contact_pred_forces - contact_gt_forces) ** 2)
    rmse = np.sqrt(mse)
    
    # 计算每个轴的误差
    mae_per_axis = np.mean(np.abs(contact_pred_forces - contact_gt_forces), axis=0)
    rmse_per_axis = np.sqrt(np.mean((contact_pred_forces - contact_gt_forces) ** 2, axis=0))
    
    # 计算环境力大小的误差
    pred_force_magnitude = np.linalg.norm(contact_pred_forces, axis=1)
    gt_force_magnitude = np.linalg.norm(contact_gt_forces, axis=1)
    magnitude_mae = np.mean(np.abs(pred_force_magnitude - gt_force_magnitude))
    magnitude_rmse = np.sqrt(np.mean((pred_force_magnitude - gt_force_magnitude) ** 2))
    
    # 计算环境力方向误差（余弦相似度）
    # 避免零向量导致的数值问题
    pred_norm = np.linalg.norm(contact_pred_forces, axis=1)
    gt_norm = np.linalg.norm(contact_gt_forces, axis=1)
    valid_mask = (pred_norm > 1e-6) & (gt_norm > 1e-6)
    
    if np.sum(valid_mask) > 0:
        pred_normalized = contact_pred_forces[valid_mask] / pred_norm[valid_mask][:, np.newaxis]
        gt_normalized = contact_gt_forces[valid_mask] / gt_norm[valid_mask][:, np.newaxis]
        cosine_similarities = np.sum(pred_normalized * gt_normalized, axis=1)
        cosine_similarities = np.clip(cosine_similarities, -1.0, 1.0)
        mean_cosine_similarity = np.mean(cosine_similarities)
        angular_errors = np.arccos(cosine_similarities) * 180.0 / np.pi
        mean_angular_error_deg = np.mean(angular_errors)
    else:
        mean_cosine_similarity = 0.0
        mean_angular_error_deg = 0.0
    
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
        'total_points': len(env_contact_mask),
        'valid_direction_points': np.sum(valid_mask) if np.sum(valid_mask) > 0 else 0
    }
    
    print(f"\n=== Environment Force Regression Results ===")
    print(f"Environment contact points used: {np.sum(contact_indices)} / {len(env_contact_mask)}")
    print(f"Overall MAE: {mae:.6f}")
    print(f"Overall RMSE: {rmse:.6f}")
    print(f"MAE per axis (X,Y,Z): [{mae_per_axis[0]:.6f}, {mae_per_axis[1]:.6f}, {mae_per_axis[2]:.6f}]")
    print(f"RMSE per axis (X,Y,Z): [{rmse_per_axis[0]:.6f}, {rmse_per_axis[1]:.6f}, {rmse_per_axis[2]:.6f}]")
    print(f"Environment force magnitude MAE: {magnitude_mae:.6f}")
    print(f"Environment force magnitude RMSE: {magnitude_rmse:.6f}")
    print(f"Mean cosine similarity: {mean_cosine_similarity:.4f}")
    print(f"Mean angular error: {mean_angular_error_deg:.2f} degrees")
    
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
    from torch.utils.data import SequentialSampler
    sampler_val = SequentialSampler(dataset_val)
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
                model.load_state_dict(checkpoint['model'], strict=True)
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=True)
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            else:
                # 如果没有找到标准键，尝试直接加载
                model.load_state_dict(checkpoint, strict=True)
            
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
    
    # 提取环境力和环境接触数据
    all_env_contact_probs = eval_data['env_contact_probs']
    all_env_contact_labels = eval_data['env_contact_labels']
    all_env_force_predictions = eval_data['env_force_predictions']
    all_env_force_ground_truth = eval_data['env_force_ground_truth']
    all_env_contact_mask = eval_data['env_contact_mask']
    
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
    
    # 3. 环境接触分类评估
    env_classification_results = None
    env_optimal_threshold_results = None
    if len(all_env_contact_probs) > 0:
        print(f"\n3. 环境接触分类评估")
        env_classification_results = evaluate_environment_contact_classification(
            all_env_contact_probs, all_env_contact_labels, threshold=0.5
        )
        
        # 3.1 分析环境接触的最优阈值（解决类别不平衡问题）
        print(f"\n3.1 环境接触最优阈值分析")
        env_optimal_threshold_results = analyze_optimal_threshold_for_imbalanced_data(
            all_env_contact_probs, all_env_contact_labels, metric='f1'
        )
        
        # 使用最优阈值重新评估
        if env_optimal_threshold_results['best_threshold'] != 0.5:
            print(f"\n3.2 使用最优阈值 {env_optimal_threshold_results['best_threshold']:.2f} 重新评估")
            env_classification_results_optimal = evaluate_environment_contact_classification(
                all_env_contact_probs, all_env_contact_labels, 
                threshold=env_optimal_threshold_results['best_threshold']
            )
            # 将最优阈值结果也保存
            env_classification_results['optimal_threshold_results'] = env_classification_results_optimal
    else:
        print(f"\n3. 环境接触分类评估 - 跳过（无数据）")
    
    # 4. 环境力回归评估
    env_regression_results = None
    if len(all_env_force_predictions) > 0:
        print(f"\n4. 环境力回归评估")
        env_regression_results = evaluate_environment_force_regression(
            all_env_force_predictions, all_env_force_ground_truth, all_env_contact_mask
        )
    else:
        print(f"\n4. 环境力回归评估 - 跳过（无数据）")
    
    # 5. 阈值分析
    print(f"\n5. 阈值分析")
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
    
    # 添加环境力和环境接触的结果（如果有数据）
    if env_classification_results is not None:
        comprehensive_results['environment_contact_classification'] = env_classification_results
    
    if env_regression_results is not None:
        comprehensive_results['environment_force_regression'] = env_regression_results
    
    # 添加环境最优阈值分析结果
    if env_optimal_threshold_results is not None:
        comprehensive_results['environment_optimal_threshold_analysis'] = env_optimal_threshold_results
    
    # 添加环境数据摘要
    if len(all_env_contact_probs) > 0:
        comprehensive_results['environment_data_summary'] = {
            'total_env_contact_samples': len(all_env_contact_probs),
            'total_env_force_samples': len(all_env_force_predictions),
            'env_contact_points': int(np.sum(all_env_contact_mask == 1)),
            'positive_env_contact_ratio': float(np.mean(all_env_contact_labels)),
            'env_contact_prob_range': [float(all_env_contact_probs.min()), float(all_env_contact_probs.max())],
            'env_force_pred_range': [float(all_env_force_predictions.min()), float(all_env_force_predictions.max())],
            'env_force_gt_range': [float(all_env_force_ground_truth.min()), float(all_env_force_ground_truth.max())]
        }
    else:
        comprehensive_results['environment_data_summary'] = {
            'total_env_contact_samples': 0,
            'total_env_force_samples': 0,
            'env_contact_points': 0,
            'positive_env_contact_ratio': 0.0,
            'message': 'No environment force/contact data found'
        }
    
    # 保存结果 - 转换numpy类型以支持JSON序列化
    results_file = os.path.join(args.output_dir, 'comprehensive_evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(convert_numpy_types(comprehensive_results), f, indent=2)
    
    save_results(results, best_threshold, best_f1, average_precision, args.output_dir)
    
    print(f"\n=== 评估总结 ===")
    print(f"物体间接触分类F1分数: {classification_results['f1_score']:.4f}")
    print(f"物体间接触分类准确率: {classification_results['accuracy']:.4f}")
    print(f"物体间力回归MAE: {regression_results['mae']:.6f}")
    print(f"物体间力回归RMSE: {regression_results['rmse']:.6f}")
    print(f"物体间力方向余弦相似度: {regression_results['mean_cosine_similarity']:.4f}")
    
    # 添加环境力评估总结
    if env_classification_results is not None and env_regression_results is not None:
        print(f"\n--- 环境力评估总结 ---")
        print(f"环境接触分类F1分数: {env_classification_results['f1_score']:.4f}")
        print(f"环境接触分类准确率: {env_classification_results['accuracy']:.4f}")
        print(f"环境力回归MAE: {env_regression_results['mae']:.6f}")
        print(f"环境力回归RMSE: {env_regression_results['rmse']:.6f}")
        print(f"环境力方向余弦相似度: {env_regression_results['mean_cosine_similarity']:.4f}")
        print(f"环境接触点数量: {env_regression_results['contact_points']}/{env_regression_results['total_points']}")
    else:
        print(f"\n--- 环境力评估总结 ---")
        print(f"未找到环境力数据或评估被跳过")
    
    print(f"\n推荐阈值: {best_threshold:.2f}")
    print(f"最佳F1分数: {best_f1:.4f}")
    print(f"平均精确率 (AP): {average_precision:.4f}")
    print(f"\n综合结果保存在: {results_file}")
    print(f"阈值分析结果保存在: {args.output_dir}")
    print("综合评估完成!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PoET阈值分析评估脚本', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)