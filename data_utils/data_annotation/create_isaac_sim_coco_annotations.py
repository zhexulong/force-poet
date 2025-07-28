#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Create COCO-style annotations for Isaac Sim dataset
# ------------------------------------------------------------------------

import json
import os
import glob
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

# YCB object mapping - maps category IDs to names
YCB_CATEGORIES = {
    1: "002_master_chef_can",
    2: "003_cracker_box",
    3: "004_sugar_box",
    4: "005_tomato_soup_can",
    5: "006_mustard_bottle",
    6: "007_tuna_fish_can",
    7: "008_pudding_box",
    8: "009_gelatin_box",
    9: "010_potted_meat_can",
    10: "011_banana",
    11: "019_pitcher_base",
    12: "021_bleach_cleanser",
    13: "024_bowl",
    14: "025_mug",
    15: "035_power_drill",
    16: "036_wood_block",
    17: "037_scissors",
    18: "040_large_marker",
    19: "051_large_clamp",
    20: "052_extra_large_clamp",
    21: "061_foam_brick",
}

def create_coco_annotations(dataset_dir, split):
    """Create COCO-style annotations for the given dataset split."""
    
    split_dir = os.path.join(dataset_dir, split)
    if not os.path.exists(split_dir):
        logger.warning(f"Split directory {split_dir} does not exist")
        return
    
    # Initialize COCO annotation structure
    coco_data = {
        "info": {
            "description": f"Isaac Sim Dataset - {split} split",
            "version": "1.0",
            "year": 2024,
            "contributor": "Isaac Sim to PoET Converter",
            "date_created": "2024"
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "categories": [],
        "images": [],
        "annotations": []
    }
    
    # Add categories
    for cat_id, cat_name in YCB_CATEGORIES.items():
        coco_data["categories"].append({
            "id": cat_id,
            "name": cat_name,
            "supercategory": "object"
        })
    
    # Process each scene
    scene_dirs = sorted([d for d in os.listdir(split_dir) 
                        if os.path.isdir(os.path.join(split_dir, d))])
    
    image_id = 0
    annotation_id = 0
    
    for scene_name in tqdm(scene_dirs, desc=f"Processing {split} scenes"):
        scene_path = os.path.join(split_dir, scene_name)
        
        # Load scene ground truth
        scene_gt_path = os.path.join(scene_path, 'scene_gt.json')
        scene_camera_path = os.path.join(scene_path, 'scene_camera.json')
        
        if not os.path.exists(scene_gt_path) or not os.path.exists(scene_camera_path):
            logger.warning(f"Missing annotation files in {scene_path}")
            continue
        
        with open(scene_gt_path, 'r') as f:
            scene_gt = json.load(f)
        
        with open(scene_camera_path, 'r') as f:
            scene_camera = json.load(f)
        
        # Process each image in the scene
        rgb_dir = os.path.join(scene_path, 'rgb')
        if not os.path.exists(rgb_dir):
            continue
        
        image_files = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
        
        for img_file in image_files:
            img_name = os.path.basename(img_file)
            img_idx = img_name.split('.')[0]  # e.g., "000000" from "000000.png"
            img_idx_int = str(int(img_idx))  # Convert "000000" to "0" for JSON key lookup
            
            # Get image dimensions
            try:
                with Image.open(img_file) as img:
                    width, height = img.size
            except Exception as e:
                logger.error(f"Error reading image {img_file}: {e}")
                continue
            
            # Add image info
            image_info = {
                "id": image_id,
                "file_name": f"{split}/{scene_name}/rgb/{img_name}",
                "width": width,
                "height": height,
                "license": 1,
                "scene_id": int(scene_name),
                "camera_id": int(img_idx)
            }
            
            # Add camera intrinsics if available
            if img_idx_int in scene_camera:
                cam_data = scene_camera[img_idx_int]
                image_info["camera_intrinsics"] = cam_data.get("cam_K", [])
                image_info["depth_scale"] = cam_data.get("depth_scale", 1000.0)
            
            coco_data["images"].append(image_info)
            
            # Add annotations for this image
            if img_idx_int in scene_gt:
                for obj_annotation in scene_gt[img_idx_int]:
                    obj_id = obj_annotation["obj_id"]
                    bbox = obj_annotation["obj_bb"]  # [x, y, width, height]
                    
                    # Calculate area
                    area = bbox[2] * bbox[3]
                    
                    # Create COCO annotation
                    annotation = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": obj_id,
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0,
                        "segmentation": [],  # We have masks but not polygon segmentation
                        
                        # PoET-specific fields
                        "camera_pose": {
                            "rotation": obj_annotation["cam_R_m2c"],
                            "translation": obj_annotation["cam_t_m2c"]
                        },
                        "object_pose": {
                            "rotation": obj_annotation["cam_R_m2c"],
                            "position": obj_annotation["cam_t_m2c"]
                        }
                    }
                    
                    coco_data["annotations"].append(annotation)
                    annotation_id += 1
            
            image_id += 1
    
    # Create annotations directory
    annotations_dir = os.path.join(dataset_dir, 'annotations')
    os.makedirs(annotations_dir, exist_ok=True)
    
    # Save COCO annotations
    output_file = os.path.join(annotations_dir, f'{split}.json')
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    logger.info(f"Created COCO annotations for {split}: {output_file}")
    logger.info(f"  - {len(coco_data['images'])} images")
    logger.info(f"  - {len(coco_data['annotations'])} annotations")
    logger.info(f"  - {len(coco_data['categories'])} categories")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create COCO annotations for Isaac Sim dataset')
    parser.add_argument('dataset_dir', type=str, help='Dataset directory')
    parser.add_argument('split', type=str, choices=['train', 'val', 'test'], help='Dataset split')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    create_coco_annotations(args.dataset_dir, args.split)
