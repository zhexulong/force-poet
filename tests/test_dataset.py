#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Isaac Sim数据集是否符合PoET要求
"""

import sys
import os
sys.path.append('/data/gplong/force_map_project/w-poet/poet')

import torch
import json
from pathlib import Path
from data_utils.pose_dataset import build
import argparse

class Args:
    """模拟命令行参数"""
    def __init__(self):
        # 使用原始数据集路径作为根目录，标注文件在子目录中
        self.dataset_path = '/data/gplong/force_map_project/isaac_sim_poet_dataset_new'
        self.synt_background = None
        self.rgb_augmentation = False
        self.grayscale = False
        self.bbox_mode = 'gt'
        self.jitter_probability = 0.5
        self.cache_mode = False

def test_dataset_loading():
    """测试数据集加载"""
    print("开始测试Isaac Sim数据集...")
    
    args = Args()
    
    # 检查数据路径是否存在
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"错误：数据集路径不存在 {dataset_path}")
        return False
    
    # 检查必要文件
    train_json = dataset_path / 'annotations' / 'train.json'
    val_json = dataset_path / 'annotations' / 'val.json'
    classes_json = dataset_path / 'annotations' / 'classes.json'
    
    for file_path in [train_json, val_json, classes_json]:
        if not file_path.exists():
            print(f"错误：缺少文件 {file_path}")
            return False
        print(f"✓ 找到文件: {file_path}")
    
    try:
        # 加载训练数据集
        print("\n加载训练数据集...")
        train_dataset = build('train', args)
        print(f"✓ 训练数据集加载成功，包含 {len(train_dataset)} 个样本")
        
        # 加载验证数据集
        print("\n加载验证数据集...")
        val_dataset = build('val', args)
        print(f"✓ 验证数据集加载成功，包含 {len(val_dataset)} 个样本")
        
        # 测试第一个样本
        print("\n测试第一个训练样本...")
        img, target = train_dataset[0]
        
        print(f"图像形状: {img.shape}")
        print(f"图像类型: {type(img)}")
        
        # 检查target字段
        required_fields = ['boxes', 'labels', 'image_id']
        optional_fields = ['relative_position', 'relative_quaternions', 'relative_rotation', 'intrinsics']
        
        print("\n检查target字段:")
        for field in required_fields:
            if field in target:
                print(f"✓ {field}: {target[field].shape if hasattr(target[field], 'shape') else type(target[field])}")
            else:
                print(f"✗ 缺少必需字段: {field}")
                return False
        
        for field in optional_fields:
            if field in target:
                print(f"✓ {field}: {target[field].shape if hasattr(target[field], 'shape') else type(target[field])}")
            else:
                print(f"- 可选字段未找到: {field}")
        
        # 检查姿态数据
        if 'relative_position' in target and 'relative_rotation' in target:
            print(f"\n姿态数据检查:")
            print(f"相对位置: {target['relative_position']}")
            print(f"相对旋转矩阵形状: {target['relative_rotation'].shape}")
            print(f"边界框: {target['boxes']}")
            print(f"类别标签: {target['labels']}")
        
        print("\n✓ 数据集测试通过！")
        return True
        
    except Exception as e:
        print(f"错误：数据集加载失败 - {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def check_data_statistics():
    """检查数据统计信息"""
    print("\n=== 数据统计信息 ===")
    
    # 加载类别信息
    classes_file = '/data/gplong/force_map_project/isaac_sim_poet_dataset_new/annotations/classes.json'
    with open(classes_file, 'r') as f:
        classes = json.load(f)
    
    print(f"类别数量: {len(classes)}")
    print("类别列表:")
    for class_id, class_name in classes.items():
        print(f"  {class_id}: {class_name}")
    
    # 加载训练标注统计
    train_file = '/data/gplong/force_map_project/isaac_sim_poet_dataset_new/annotations/train.json'
    with open(train_file, 'r') as f:
        train_data = json.load(f)
    
    print(f"\n训练集图像数量: {len(train_data['images'])}")
    print(f"训练集标注数量: {len(train_data['annotations'])}")
    
    # 统计每个类别的标注数量
    category_counts = {}
    for ann in train_data['annotations']:
        cat_id = ann['category_id']
        category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
    
    print("\n各类别标注数量:")
    for cat_id, count in sorted(category_counts.items()):
        class_name = classes.get(str(cat_id), f"unknown_{cat_id}")
        print(f"  {class_name}: {count}")

if __name__ == '__main__':
    print("Isaac Sim数据集PoET兼容性测试")
    print("=" * 50)
    
    # 测试数据集加载
    success = test_dataset_loading()
    
    if success:
        # 检查数据统计
        check_data_statistics()
        print("\n🎉 所有测试通过！数据集符合PoET要求。")
    else:
        print("\n❌ 测试失败！数据集不符合PoET要求。")
        sys.exit(1)