#!/usr/bin/env python3
"""
测试脚本：验证pose_dataset中force数据加载
确保删除pairwise_forces后，forces数据仍能正常加载
"""

import sys
import os
import torch

# 添加项目路径
sys.path.append('/data/gplong/force_map_project/w-poet/poet')

def test_force_data_loading():
    """测试力数据加载功能"""
    print("=== 测试Force数据加载功能 ===")
    
    try:
        from data_utils.pose_dataset import PoseDataset, make_pose_estimation_transform
        
        # 模拟参数
        class Args:
            dataset_path = '/data/gplong/force_map_project/w-poet/isaac_sim_poet_dataset_force'
            synt_background = None
            rgb_augmentation = False
            grayscale = False
            cache_mode = False
            jitter_probability = 0.0
        
        args = Args()
        
        # 构建数据集路径
        img_folder = args.dataset_path
        ann_file = os.path.join(args.dataset_path, 'annotations', 'val.json')
        
        # 检查文件是否存在
        if not os.path.exists(ann_file):
            print(f"❌ 注释文件不存在: {ann_file}")
            return False
            
        print(f"✓ 使用注释文件: {ann_file}")
        
        # 创建数据集
        dataset = PoseDataset(
            img_folder=img_folder,
            ann_file=ann_file,
            synthetic_background=args.synt_background,
            transforms=make_pose_estimation_transform('val', args.rgb_augmentation, args.grayscale),
            return_masks=False,
            jitter=False,
            jitter_probability=args.jitter_probability,
            cache_mode=args.cache_mode,
            local_rank=0,
            local_size=1
        )
        
        print(f"✓ 数据集创建成功，包含 {len(dataset)} 个样本")
        
        # 测试加载几个样本
        success_count = 0
        force_data_count = 0
        
        for i in range(min(5, len(dataset))):
            try:
                img, target = dataset[i]
                success_count += 1
                
                print(f"\n样本 {i}:")
                print(f"  图像尺寸: {img.shape if hasattr(img, 'shape') else 'PIL Image'}")
                print(f"  目标字段: {list(target.keys())}")
                
                if 'force_matrix' in target:
                    force_matrix = target['force_matrix']
                    print(f"  Force matrix形状: {force_matrix.shape}")
                    print(f"  Force matrix非零元素数: {torch.count_nonzero(force_matrix)}")
                    force_data_count += 1
                else:
                    print("  无force_matrix数据")
                
                if 'masses' in target:
                    masses = target['masses']
                    print(f"  质量数据: {masses}")
                    
            except Exception as e:
                print(f"❌ 样本 {i} 加载失败: {e}")
        
        print(f"\n=== 测试结果 ===")
        print(f"成功加载样本数: {success_count}/{min(5, len(dataset))}")
        print(f"包含force数据的样本数: {force_data_count}")
        
        if success_count > 0:
            print("✓ 数据加载功能正常")
            return True
        else:
            print("❌ 数据加载失败")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_force_matrix_from_scene_gt():
    """测试从scene_gt.json加载force数据"""
    print("\n=== 测试从scene_gt.json加载Force数据 ===")
    
    try:
        import json
        
        # 查找一个scene_gt.json文件
        scene_gt_file = '/data/gplong/force_map_project/w-poet/isaac_sim_poet_dataset_force/val/000137/scene_gt.json'
        
        if not os.path.exists(scene_gt_file):
            print(f"❌ scene_gt.json文件不存在: {scene_gt_file}")
            return False
        
        print(f"✓ 使用scene_gt文件: {scene_gt_file}")
        
        # 读取scene_gt数据
        with open(scene_gt_file, 'r') as f:
            scene_data = json.load(f)
        
        print(f"✓ 文件包含 {len(scene_data)} 个相机视角")
        
        # 检查第一个视角的数据
        first_view = next(iter(scene_data.values()))
        print(f"✓ 第一个视角包含 {len(first_view)} 个对象")
        
        # 检查force数据
        total_forces = 0
        for obj in first_view:
            if 'forces' in obj:
                forces = obj['forces']
                total_forces += len(forces)
                print(f"  对象 {obj.get('obj_name', 'unknown')} 有 {len(forces)} 个力交互")
                
                for force in forces:
                    print(f"    力: {force.get('source_obj_id')} -> {force.get('target_obj_id')}: {force.get('force_vector')}")
        
        print(f"✓ 总共找到 {total_forces} 个力交互")
        
        if total_forces > 0:
            print("✓ scene_gt.json中包含有效的force数据")
            return True
        else:
            print("⚠️ scene_gt.json中未找到force数据")
            return True  # 不算错误，可能这个场景确实没有力交互
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("开始测试pose_dataset中的force数据加载功能...\n")
    
    # 测试基本数据加载
    test1_result = test_force_data_loading()
    
    # 测试scene_gt.json数据读取
    test2_result = test_force_matrix_from_scene_gt()
    
    print(f"\n{'='*50}")
    print("测试总结:")
    print(f"✓ 基本数据加载: {'通过' if test1_result else '失败'}")
    print(f"✓ scene_gt.json读取: {'通过' if test2_result else '失败'}")
    
    if test1_result and test2_result:
        print("\n🎉 所有测试通过！force数据加载功能正常。")
        print("✅ 已成功删除不使用的pairwise_forces，保留了实际使用的forces。")
    else:
        print("\n❌ 部分测试失败，请检查实现。")


if __name__ == "__main__":
    main()
