#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from data_utils import build_dataset
import argparse

# Create simple args object
class Args:
    def __init__(self):
        self.dataset_path = "../isaac_sim_poet_dataset_force"
        self.dataset = "isaac_sim"
        self.train_set = "train"
        self.synt_background = None
        self.rgb_augmentation = False
        self.grayscale = False
        self.bbox_mode = 'gt'
        self.jitter_probability = 0.0
        self.cache_mode = False
        self.std = 0.02

def test_dataset_loading():
    args = Args()
    
    # Create dataset using build_dataset function
    dataset = build_dataset(image_set=args.train_set, args=args)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test first few samples
    for i in range(min(3, len(dataset))):
        print(f"\n--- Sample {i} ---")
        sample, target = dataset[i]
        
        print(f"Target keys: {list(target.keys())}")
        
        if 'force_matrix' in target:
            force_matrix = target['force_matrix']
            print(f"Force matrix shape: {force_matrix.shape}")
            nonzero_count = torch.norm(force_matrix, dim=-1) > 1e-6
            print(f"Non-zero force entries: {nonzero_count.sum().item()}")
            
            # Show some non-zero forces
            if nonzero_count.sum() > 0:
                print("Sample forces:")
                for r in range(min(force_matrix.shape[0], 5)):
                    for c in range(min(force_matrix.shape[1], 5)):
                        if torch.norm(force_matrix[r, c]) > 1e-6:
                            print(f"  [{r}, {c}] = {force_matrix[r, c].tolist()}")
        else:
            print("No force_matrix in target")

if __name__ == "__main__":
    test_dataset_loading()
