#!/usr/bin/env python3
"""
清理标注文件，移除不存在的图像文件对应的标注
"""

import json
import os
from pathlib import Path

def clean_annotations(dataset_path, annotation_file, output_file):
    """
    清理标注文件，移除不存在的图像文件
    """
    print(f"正在清理标注文件: {annotation_file}")
    
    # 读取原始标注文件
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    original_images = len(data['images'])
    original_annotations = len(data['annotations'])
    
    # 检查图像文件是否存在
    valid_images = []
    valid_image_ids = set()
    
    for img in data['images']:
        img_path = os.path.join(dataset_path, img['file_name'])
        if os.path.exists(img_path):
            valid_images.append(img)
            valid_image_ids.add(img['id'])
        else:
            print(f"图像文件不存在: {img_path}")
    
    # 过滤标注，只保留有效图像的标注
    valid_annotations = []
    for ann in data['annotations']:
        if ann['image_id'] in valid_image_ids:
            valid_annotations.append(ann)
    
    # 更新数据
    data['images'] = valid_images
    data['annotations'] = valid_annotations
    
    # 保存清理后的标注文件
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"清理完成:")
    print(f"  原始图像数量: {original_images}")
    print(f"  有效图像数量: {len(valid_images)}")
    print(f"  原始标注数量: {original_annotations}")
    print(f"  有效标注数量: {len(valid_annotations)}")
    print(f"  清理后的文件保存到: {output_file}")

def main():
    dataset_path = "/data/gplong/force_map_project/isaac_sim_poet_dataset_new"
    annotations_dir = os.path.join(dataset_path, "annotations")
    
    # 清理训练集标注
    train_file = os.path.join(annotations_dir, "train.json")
    train_clean_file = os.path.join(annotations_dir, "train_clean.json")
    clean_annotations(dataset_path, train_file, train_clean_file)
    
    # 清理验证集标注
    val_file = os.path.join(annotations_dir, "val.json")
    val_clean_file = os.path.join(annotations_dir, "val_clean.json")
    clean_annotations(dataset_path, val_file, val_clean_file)
    
    print("\n所有标注文件清理完成！")

if __name__ == "__main__":
    main()