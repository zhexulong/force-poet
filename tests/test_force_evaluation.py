#!/usr/bin/env python3
"""
测试脚本：验证力预测评估功能
"""

import numpy as np
import sys
import os
import tempfile
import shutil

# 添加项目路径
sys.path.append('/data/gplong/force_map_project/w-poet/poet')

from evaluation_tools.pose_evaluator import PoseEvaluator


def create_test_evaluator():
    """创建测试用的pose evaluator"""
    # 模拟模型和类别数据
    models = {
        'test_class_1': {'pts': np.random.rand(100, 3)},
        'test_class_2': {'pts': np.random.rand(100, 3)}
    }
    
    classes = {'1': 'test_class_1', '2': 'test_class_2'}
    model_info = {
        'test_class_1': {'diameter': 0.1},
        'test_class_2': {'diameter': 0.15}
    }
    model_symmetry = {
        'test_class_1': False,
        'test_class_2': True
    }
    
    evaluator = PoseEvaluator(models, classes, model_info, model_symmetry)
    return evaluator


def test_force_matrix_evaluation():
    """测试force matrix评估功能"""
    print("=== 测试Force Matrix评估功能 ===")
    
    evaluator = create_test_evaluator()
    
    # 创建模拟的力矩阵数据
    n_samples = 5
    matrix_size = 3
    
    for cls in evaluator.classes:
        for i in range(n_samples):
            # 创建随机的预测和真实力矩阵
            pred_matrix = np.random.randn(matrix_size, matrix_size, 3) * 0.1
            gt_matrix = np.random.randn(matrix_size, matrix_size, 3) * 0.1
            
            # 添加一些明显的力关系（非零值）
            if i < 2:  # 前两个样本有明显的力关系
                gt_matrix[0, 1] = [1.0, 0.5, -0.3]  # 对象0对对象1的力
                gt_matrix[1, 0] = [-1.0, -0.5, 0.3]  # 牛顿第三定律
                pred_matrix[0, 1] = [0.9, 0.4, -0.25]  # 近似但不完全正确的预测
                pred_matrix[1, 0] = [-0.9, -0.4, 0.25]
            
            evaluator.force_matrices_pred[cls].append(pred_matrix)
            evaluator.force_matrices_gt[cls].append(gt_matrix)
    
    # 创建临时输出目录
    temp_dir = tempfile.mkdtemp()
    try:
        # 测试详细的force matrix评估
        print("测试详细Force Matrix评估...")
        results = evaluator.evaluate_force_matrix_prediction(temp_dir, epoch=0)
        
        print("评估结果:")
        if "overall" in results:
            overall = results["overall"]
            print(f"  整体Vector MSE: {overall['vector_mse']:.6f}")
            print(f"  整体Vector MAE: {overall['vector_mae']:.6f}")
            print(f"  整体方向准确性: {overall['direction_accuracy']:.6f}")
            print(f"  整体检测准确性: {overall['detection_accuracy']:.6f}")
            print(f"  整体精确度: {overall['precision']:.6f}")
            print(f"  整体召回率: {overall['recall']:.6f}")
            print(f"  整体F1分数: {overall['f1_score']:.6f}")
        
        # 测试简化的平均误差计算
        print("\n测试简化Force Matrix误差计算...")
        avg_error = evaluator.calculate_class_avg_force_matrix_error(temp_dir, epoch=0)
        print(f"平均Force Matrix误差: {avg_error:.6f}")
        
        print("✓ Force Matrix评估功能测试通过")
        
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir)


def test_traditional_force_evaluation():
    """测试传统的力评估功能"""
    print("\n=== 测试传统Force评估功能 ===")
    
    evaluator = create_test_evaluator()
    
    # 创建模拟的简单力数据
    n_samples = 10
    
    for cls in evaluator.classes:
        for i in range(n_samples):
            # 创建随机的预测和真实力值
            pred_force = np.random.randn() * 0.5 + 1.0  # 围绕1.0的力值
            gt_force = np.random.randn() * 0.3 + 1.0
            
            evaluator.forces_pred[cls].append(pred_force)
            evaluator.forces_gt[cls].append(gt_force)
    
    # 创建临时输出目录
    temp_dir = tempfile.mkdtemp()
    try:
        # 测试传统force评估
        print("测试传统Force评估...")
        mae, rmse = evaluator.evaluate_force_prediction(temp_dir, epoch=0)
        print(f"Force MAE: {mae:.6f}")
        print(f"Force RMSE: {rmse:.6f}")
        
        # 测试平均力误差
        avg_error = evaluator.calculate_class_avg_force_error(temp_dir, epoch=0)
        print(f"平均Force误差: {avg_error:.6f}")
        
        print("✓ 传统Force评估功能测试通过")
        
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir)


def main():
    """主测试函数"""
    print("开始测试力预测评估功能...\n")
    
    try:
        # 测试force matrix评估
        test_force_matrix_evaluation()
        
        # 测试传统force评估
        test_traditional_force_evaluation()
        
        print("\n🎉 所有测试通过！力预测评估功能工作正常。")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
