#!/usr/bin/env python3

import sys
import os
import json
import torch
import numpy as np
from pathlib import Path

def test_force_data_in_files():
    """Test if force data exists in the raw dataset files"""
    dataset_path = Path("../isaac_sim_poet_dataset_force")
    
    # Check a few train samples
    train_dir = dataset_path / "train"
    sample_dirs = list(train_dir.glob("*"))[:5]  # First 5 samples
    
    print(f"Testing {len(sample_dirs)} samples from {train_dir}")
    
    for sample_dir in sample_dirs:
        print(f"\n--- Sample {sample_dir.name} ---")
        
        # Check scene_gt.json for force data
        scene_gt_file = sample_dir / "scene_gt.json"
        if scene_gt_file.exists():
            with open(scene_gt_file, 'r') as f:
                scene_data = json.load(f)
            
            print(f"Views in scene: {list(scene_data.keys())}")
            
            # Check first view for force data
            if '0' in scene_data:
                objects = scene_data['0']
                print(f"Objects in view 0: {len(objects)}")
                
                force_count = 0
                for obj in objects:
                    if 'forces' in obj and obj['forces']:
                        force_count += len(obj['forces'])
                        print(f"Object {obj.get('obj_id', 'N/A')} has {len(obj['forces'])} forces")
                        for force in obj['forces'][:2]:  # Show first 2 forces
                            print(f"  Force: {force}")
                
                print(f"Total force interactions in view 0: {force_count}")
        else:
            print("No scene_gt.json found")

def test_dataset_build():
    """Test if our dataset build process works"""
    try:
        # Minimal imports for testing
        sys.path.append('.')
        from data_utils.pose_dataset import PoseDataset, make_pose_estimation_transform
        from util.misc import get_local_rank, get_local_size
        
        dataset_path = Path("../isaac_sim_poet_dataset_force")
        img_folder = dataset_path
        ann_file = dataset_path / "annotations" / "train.json"
        
        print(f"Checking annotation file: {ann_file}")
        print(f"Annotation file exists: {ann_file.exists()}")
        
        if ann_file.exists():
            # Try to create dataset
            dataset = PoseDataset(
                img_folder, ann_file, 
                synthetic_background=None,
                transforms=make_pose_estimation_transform("train", False, False),
                return_masks=False, 
                jitter=False,
                cache_mode=False, 
                local_rank=0, 
                local_size=1
            )
            
            print(f"Dataset created successfully with {len(dataset)} samples")
            
            # Test first sample
            if len(dataset) > 0:
                sample, target = dataset[0]
                print(f"First sample target keys: {list(target.keys())}")
                
                if 'force_matrix' in target:
                    force_matrix = target['force_matrix']
                    print(f"Force matrix shape: {force_matrix.shape}")
                    nonzero_count = torch.norm(force_matrix, dim=-1) > 1e-6
                    print(f"Non-zero force entries: {nonzero_count.sum().item()}")
                else:
                    print("No force_matrix in target")
        else:
            print("Annotation file not found - using direct JSON approach")
            
    except Exception as e:
        print(f"Error creating dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=== Testing raw force data in files ===")
    test_force_data_in_files()
    
    print("\n=== Testing dataset build process ===")
    test_dataset_build()
