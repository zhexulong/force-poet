#!/usr/bin/env python3
"""
物理约束损失函数演示脚本

本脚本演示了如何使用新实现的物理约束损失函数：
1. 归一化掩码对称损失 (loss_force_symmetry) - 强制执行牛顿第三定律
2. 物理一致性损失 (loss_force_consistency) - 强制执行牛顿第一定律
"""

import torch
import numpy as np
import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'poet'))

from poet.models.pose_estimation_transformer import SetCriterion
from poet.models.matcher import PoseMatcher


def create_symmetric_force_matrix(batch_size, num_objects):
    """创建一个满足牛顿第三定律的对称力矩阵"""
    force_matrix = torch.zeros(batch_size, num_objects, num_objects, 3)
    
    # 为每一对物体创建对称的力
    for b in range(batch_size):
        for i in range(num_objects):
            for j in range(i + 1, num_objects):
                # 随机生成一个力向量
                force_ij = torch.randn(3) * 2.0  # 随机力，范围 [-2, 2]
                
                # 设置 F_ij = force_ij, F_ji = -force_ij (牛顿第三定律)
                force_matrix[b, i, j] = force_ij
                force_matrix[b, j, i] = -force_ij
    
    return force_matrix


def create_asymmetric_force_matrix(batch_size, num_objects):
    """创建一个违反牛顿第三定律的非对称力矩阵"""
    force_matrix = torch.randn(batch_size, num_objects, num_objects, 3)
    
    # 将对角线设为零（物体不对自己施力）
    for b in range(batch_size):
        for i in range(num_objects):
            force_matrix[b, i, i] = torch.zeros(3)
    
    return force_matrix


def demo_physics_constraints():
    """演示物理约束损失函数的效果"""
    print("=" * 60)
    print("物理约束损失函数演示")
    print("=" * 60)
    
    # 设置参数
    batch_size = 2
    num_queries = 5
    num_objects = 4
    
    # 创建匹配器和损失函数
    matcher = PoseMatcher(cost_class=1, cost_bbox=1, bbox_mode='gt', class_mode='agnostic')
    weight_dict = {
        'loss_force_matrix': 1.0,
        'loss_force_symmetry': 0.5,
        'loss_force_consistency': 0.3
    }
    losses = ['force_matrix', 'force_symmetry', 'force_consistency']
    
    criterion = SetCriterion(matcher, weight_dict, losses, hard_negative_ratio=0.2)
    
    # 创建目标数据
    targets = []
    for i in range(batch_size):
        target = {
            'force_matrix': torch.randn(num_objects, num_objects, 3),
            'labels': torch.randint(0, 2, (num_objects,)),
            'boxes': torch.randn(num_objects, 4)
        }
        targets.append(target)
    
    # 创建匹配索引
    indices = []
    for i in range(batch_size):
        indices.append((torch.arange(num_objects), torch.arange(num_objects)))
    
    print("\n1. 测试对称力矩阵（满足牛顿第三定律）")
    print("-" * 40)
    
    # 创建对称力矩阵
    symmetric_force = create_symmetric_force_matrix(batch_size, num_queries)
    
    outputs_symmetric = {
        'pred_force_matrix': symmetric_force,
        'pred_boxes': torch.randn(batch_size, num_queries, 4),
        'pred_classes': torch.randint(0, 2, (batch_size, num_queries))
    }
    
    # 计算损失
    symmetry_loss = criterion.loss_force_symmetry(outputs_symmetric, targets, indices)
    consistency_loss = criterion.loss_force_consistency(outputs_symmetric, targets, indices)
    
    print(f"对称性损失: {symmetry_loss['loss_force_symmetry'].item():.6f}")
    print(f"一致性损失: {consistency_loss['loss_force_consistency'].item():.6f}")
    
    # 验证对称性
    force_diff = symmetric_force + symmetric_force.transpose(1, 2)
    symmetry_error = torch.norm(force_diff).item()
    print(f"对称性验证 (||F + F^T||): {symmetry_error:.6f}")
    
    print("\n2. 测试非对称力矩阵（违反牛顿第三定律）")
    print("-" * 40)
    
    # 创建非对称力矩阵
    asymmetric_force = create_asymmetric_force_matrix(batch_size, num_queries)
    
    outputs_asymmetric = {
        'pred_force_matrix': asymmetric_force,
        'pred_boxes': torch.randn(batch_size, num_queries, 4),
        'pred_classes': torch.randint(0, 2, (batch_size, num_queries))
    }
    
    # 计算损失
    symmetry_loss = criterion.loss_force_symmetry(outputs_asymmetric, targets, indices)
    consistency_loss = criterion.loss_force_consistency(outputs_asymmetric, targets, indices)
    
    print(f"对称性损失: {symmetry_loss['loss_force_symmetry'].item():.6f}")
    print(f"一致性损失: {consistency_loss['loss_force_consistency'].item():.6f}")
    
    # 验证对称性
    force_diff = asymmetric_force + asymmetric_force.transpose(1, 2)
    symmetry_error = torch.norm(force_diff).item()
    print(f"对称性验证 (||F + F^T||): {symmetry_error:.6f}")
    
    print("\n3. 力平衡分析（牛顿第一定律）")
    print("-" * 40)
    
    # 分析每个物体受到的合力
    for matrix_name, force_matrix in [("对称矩阵", symmetric_force), ("非对称矩阵", asymmetric_force)]:
        print(f"\n{matrix_name}的力平衡分析:")
        
        # 计算每个物体受到的合力
        net_forces = torch.sum(force_matrix, dim=2)  # 对所有其他物体求和
        
        for b in range(batch_size):
            print(f"  批次 {b}:")
            for i in range(min(num_objects, num_queries)):
                net_force_magnitude = torch.norm(net_forces[b, i]).item()
                print(f"    物体 {i} 合力大小: {net_force_magnitude:.4f}")
    
    print("\n" + "=" * 60)
    print("总结:")
    print("- 对称性损失惩罚违反牛顿第三定律的力预测")
    print("- 一致性损失惩罚违反牛顿第一定律的力预测")
    print("- 这些约束帮助模型学习物理上合理的力相互作用")
    print("=" * 60)


if __name__ == "__main__":
    demo_physics_constraints()