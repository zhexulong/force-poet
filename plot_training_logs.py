#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练日志可视化脚本
解析train_rcnn_loss.log文件，生成loss曲线和评估数据的可视化图表
同时生成简洁的JSON格式数据供AI分析
"""

import re
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_log_file(log_path):
    """
    解析训练日志文件
    """
    training_data = defaultdict(list)
    evaluation_data = defaultdict(list)
    epoch_data = []
    
    current_epoch = None
    epoch_losses = defaultdict(list)
    
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # 解析epoch开始
        epoch_match = re.search(r'Epoch: \[(\d+)\]', line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
        
        # 解析训练数据
        if current_epoch is not None and 'loss:' in line and 'eta:' in line:
            # 提取各种loss值
            loss_patterns = {
                'total_loss': r'loss: ([\d\.]+)',
                'position_loss': r'position_loss: ([\d\.]+)',
                'rotation_loss': r'rotation_loss: ([\d\.]+)',
                'force_loss': r'force_loss: ([\d\.]+)',
                'loss_trans': r'loss_trans: ([\d\.]+)',
                'loss_rot': r'loss_rot: ([\d\.]+)',
                'loss_force_matrix': r'loss_force_matrix: ([\d\.]+)',
                'loss_force_symmetry': r'loss_force_symmetry: ([\d\.]+)',
                'loss_force_consistency': r'loss_force_consistency: ([\d\.]+)',
                'grad_norm': r'grad_norm: ([\d\.]+)'
            }
            
            batch_data = {'epoch': current_epoch}
            for key, pattern in loss_patterns.items():
                match = re.search(pattern, line)
                if match:
                    batch_data[key] = float(match.group(1))
                    epoch_losses[key].append(float(match.group(1)))
            
            if batch_data:
                training_data['batches'].append(batch_data)
        
        # 解析评估结果
        if 'Force Matrix Evaluation Results:' in line:
            eval_data = {'epoch': current_epoch}
            
            # 读取接下来的评估指标行
            j = i + 1
            while j < len(lines) and j < i + 10:  # 最多读取10行
                eval_line = lines[j].strip()
                
                eval_patterns = {
                    'avg_force_matrix_error': r'Average Force Matrix Error: ([\d\.]+)',
                    'vector_mse': r'Vector MSE: ([\d\.]+)',
                    'vector_mae': r'Vector MAE: ([\d\.]+)',
                    'direction_accuracy': r'Direction Accuracy: ([\d\.]+)',
                    'detection_accuracy': r'Detection Accuracy: ([\d\.]+)',
                    'precision': r'Precision: ([\d\.]+)',
                    'recall': r'Recall: ([\d\.]+)',
                    'f1_score': r'F1 Score: ([\d\.]+)'
                }
                
                for key, pattern in eval_patterns.items():
                    match = re.search(pattern, eval_line)
                    if match:
                        eval_data[key] = float(match.group(1))
                
                j += 1
            
            if len(eval_data) > 1:  # 确保有评估数据
                evaluation_data['epochs'].append(eval_data)
                
                # 计算该epoch的平均loss
                epoch_avg_losses = {}
                for key, values in epoch_losses.items():
                    if values:
                        epoch_avg_losses[key] = np.mean(values)
                
                # 合并epoch数据
                epoch_summary = {
                    'epoch': current_epoch,
                    'losses': epoch_avg_losses,
                    'evaluation': eval_data
                }
                epoch_data.append(epoch_summary)
                
                # 清空当前epoch的loss累积
                epoch_losses = defaultdict(list)
        
        i += 1
    
    return training_data, evaluation_data, epoch_data

def create_loss_eval_plots(epoch_data, output_dir):
    """
    Create loss and evaluation plots
    """
    if not epoch_data:
        print("No valid epoch data found")
        return
    
    epochs = [data['epoch'] for data in epoch_data]
    
    # 1. Force-related loss and evaluation metrics
    fig, axes = plt.subplots(4, 1, figsize=(12, 16))
    fig.suptitle('Force Loss and Evaluation Metrics', fontsize=16)
    
    # Force Loss
    force_losses = [data['losses'].get('force_loss', 0) for data in epoch_data]
    force_matrix_losses = [data['losses'].get('loss_force_matrix', 0) for data in epoch_data]
    axes[0].plot(epochs, force_losses, 'b-', label='Force Loss', linewidth=2)
    axes[0].plot(epochs, force_matrix_losses, 'r--', label='Force Matrix Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Force Loss Changes')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Force evaluation metrics
    vector_mse = [data['evaluation'].get('vector_mse', 0) for data in epoch_data]
    vector_mae = [data['evaluation'].get('vector_mae', 0) for data in epoch_data]
    axes[1].plot(epochs, vector_mse, 'g-', label='Vector MSE', linewidth=2)
    axes[1].plot(epochs, vector_mae, 'm--', label='Vector MAE', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Error')
    axes[1].set_title('Force Vector Error Metrics')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Force accuracy metrics
    direction_acc = [data['evaluation'].get('direction_accuracy', 0) for data in epoch_data]
    detection_acc = [data['evaluation'].get('detection_accuracy', 0) for data in epoch_data]
    axes[2].plot(epochs, direction_acc, 'c-', label='Direction Accuracy', linewidth=2)
    axes[2].plot(epochs, detection_acc, 'y--', label='Detection Accuracy', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Force Accuracy Metrics')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Precision, Recall, F1
    precision = [data['evaluation'].get('precision', 0) for data in epoch_data]
    recall = [data['evaluation'].get('recall', 0) for data in epoch_data]
    f1_score = [data['evaluation'].get('f1_score', 0) for data in epoch_data]
    axes[3].plot(epochs, precision, 'r-', label='Precision', linewidth=2)
    axes[3].plot(epochs, recall, 'g--', label='Recall', linewidth=2)
    axes[3].plot(epochs, f1_score, 'b:', label='F1 Score', linewidth=2)
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('Score')
    axes[3].set_title('Force Detection Performance Metrics')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'force_loss_eval_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Position-related loss and evaluation
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Position Loss and Evaluation Metrics', fontsize=16)
    
    position_losses = [data['losses'].get('position_loss', 0) for data in epoch_data]
    trans_losses = [data['losses'].get('loss_trans', 0) for data in epoch_data]
    axes[0].plot(epochs, position_losses, 'b-', label='Position Loss', linewidth=2)
    axes[0].plot(epochs, trans_losses, 'r--', label='Translation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Position Loss Changes')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Position evaluation metrics (using vector errors as position-related metrics)
    vector_mse = [data['evaluation'].get('vector_mse', 0) for data in epoch_data]
    vector_mae = [data['evaluation'].get('vector_mae', 0) for data in epoch_data]
    axes[1].plot(epochs, vector_mse, 'g-', label='Position Vector MSE', linewidth=2)
    axes[1].plot(epochs, vector_mae, 'm--', label='Position Vector MAE', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Error')
    axes[1].set_title('Position Evaluation Metrics')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'position_loss_eval_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Rotation-related loss and evaluation
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Rotation Loss and Evaluation Metrics', fontsize=16)
    
    rotation_losses = [data['losses'].get('rotation_loss', 0) for data in epoch_data]
    rot_losses = [data['losses'].get('loss_rot', 0) for data in epoch_data]
    axes[0].plot(epochs, rotation_losses, 'b-', label='Rotation Loss', linewidth=2)
    axes[0].plot(epochs, rot_losses, 'r--', label='Rot Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Rotation Loss Changes')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Rotation evaluation metrics (using direction accuracy as rotation-related metric)
    direction_acc = [data['evaluation'].get('direction_accuracy', 0) for data in epoch_data]
    avg_force_error = [data['evaluation'].get('avg_force_matrix_error', 0) for data in epoch_data]
    axes[1].plot(epochs, direction_acc, 'c-', label='Direction Accuracy', linewidth=2)
    axes[1].plot(epochs, avg_force_error, 'orange', label='Avg Force Matrix Error', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Metric Value')
    axes[1].set_title('Rotation Evaluation Metrics')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rotation_loss_eval_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Overall training trend
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    total_losses = [data['losses'].get('total_loss', 0) for data in epoch_data]
    grad_norms = [data['losses'].get('grad_norm', 0) for data in epoch_data]
    
    ax1 = ax
    ax2 = ax1.twinx()
    
    line1 = ax1.plot(epochs, total_losses, 'b-', label='Total Loss', linewidth=2)
    line2 = ax2.plot(epochs, grad_norms, 'r--', label='Gradient Norm', linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss', color='b')
    ax2.set_ylabel('Gradient Norm', color='r')
    ax1.set_title('Overall Training Trend')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_training_trend.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated 4 comparison charts, saved in {output_dir} directory")

def generate_simplified_json(epoch_data, output_dir):
    """
    Generate simplified JSON format data for AI analysis
    """
    simplified_data = {
        'metadata': {
            'total_epochs': len(epoch_data),
            'generated_time': datetime.now().isoformat(),
            'description': 'Simplified training log data for AI analysis'
        },
        'epoch_summary': []
    }
    
    for data in epoch_data:
        epoch_summary = {
            'epoch': data['epoch'],
            'losses': {
                'total': data['losses'].get('total_loss', 0),
                'position': data['losses'].get('position_loss', 0),
                'rotation': data['losses'].get('rotation_loss', 0),
                'force': data['losses'].get('force_loss', 0),
                'force_matrix': data['losses'].get('loss_force_matrix', 0),
                'grad_norm': data['losses'].get('grad_norm', 0)
            },
            'evaluation': {
                'force_matrix_error': data['evaluation'].get('avg_force_matrix_error', 0),
                'vector_mse': data['evaluation'].get('vector_mse', 0),
                'vector_mae': data['evaluation'].get('vector_mae', 0),
                'direction_accuracy': data['evaluation'].get('direction_accuracy', 0),
                'detection_accuracy': data['evaluation'].get('detection_accuracy', 0),
                'precision': data['evaluation'].get('precision', 0),
                'recall': data['evaluation'].get('recall', 0),
                'f1_score': data['evaluation'].get('f1_score', 0)
            }
        }
        simplified_data['epoch_summary'].append(epoch_summary)
    
    # Add statistical information
    if epoch_data:
        final_epoch = epoch_data[-1]
        simplified_data['final_performance'] = {
            'epoch': final_epoch['epoch'],
            'best_metrics': {
                'lowest_total_loss': min([d['losses'].get('total_loss', float('inf')) for d in epoch_data]),
                'lowest_force_error': min([d['evaluation'].get('avg_force_matrix_error', float('inf')) for d in epoch_data]),
                'highest_f1_score': max([d['evaluation'].get('f1_score', 0) for d in epoch_data]),
                'highest_direction_accuracy': max([d['evaluation'].get('direction_accuracy', 0) for d in epoch_data])
            }
        }
    
    # Save JSON file
    json_path = os.path.join(output_dir, 'training_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(simplified_data, f, indent=2, ensure_ascii=False)
    
    print(f"Generated simplified JSON data: {json_path}")
    return simplified_data

def main():
    # Configure paths
    log_file = '/data/gplong/force_map_project/w-poet/poet/train_rcnn_resume.log'
    output_dir = '/data/gplong/force_map_project/w-poet/poet/log_fig'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting to parse training logs...")
    training_data, evaluation_data, epoch_data = parse_log_file(log_file)
    
    print(f"Parsing completed, found {len(epoch_data)} epochs of data")
    
    if not epoch_data:
        print("Error: No valid training data found")
        return
    
    print("Generating visualization charts...")
    create_loss_eval_plots(epoch_data, output_dir)
    
    print("Generating simplified JSON data...")
    simplified_data = generate_simplified_json(epoch_data, output_dir)
    
    print("\n=== Processing Complete ===")
    print(f"Output directory: {output_dir}")
    print("Generated files:")
    print("  - force_loss_eval_comparison.png (Force loss and evaluation metrics comparison)")
    print("  - position_loss_eval_comparison.png (Position loss and evaluation metrics)")
    print("  - rotation_loss_eval_comparison.png (Rotation loss and evaluation metrics)")
    print("  - overall_training_trend.png (Overall training trend)")
    print("  - training_summary.json (Simplified training data)")
    
    # Display some statistics
    if simplified_data['epoch_summary']:
        final_metrics = simplified_data['epoch_summary'][-1]
        print(f"\nFinal Performance (Epoch {final_metrics['epoch']}):")
        print(f"  Total Loss: {final_metrics['losses']['total']:.4f}")
        print(f"  Force Matrix Error: {final_metrics['evaluation']['force_matrix_error']:.6f}")
        print(f"  F1 Score: {final_metrics['evaluation']['f1_score']:.6f}")
        print(f"  Direction Accuracy: {final_metrics['evaluation']['direction_accuracy']:.6f}")

if __name__ == '__main__':
    main()