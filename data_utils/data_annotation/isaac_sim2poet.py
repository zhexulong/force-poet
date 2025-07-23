# ------------------------------------------------------------------------
# Isaac Sim to PoET Dataset Converter
# Converts Isaac Sim dataset format to PoET-compatible COCO format
# ------------------------------------------------------------------------

import json
import os
import numpy as np
from pathlib import Path

def convert_isaac_sim_to_poet(isaac_sim_path, output_path, split='train'):
    """
    Convert Isaac Sim dataset to PoET format
    
    Args:
        isaac_sim_path: Path to Isaac Sim dataset
        output_path: Path to output PoET format annotations
        split: Dataset split ('train', 'val', 'test')
    """
    
    # Load Isaac Sim classes
    with open(os.path.join(isaac_sim_path, 'annotations', 'classes.json'), 'r') as f:
        isaac_classes = json.load(f)
    
    # Create categories for PoET format
    categories = [{'supercategory': 'background', 'id': 0, 'name': 'background'}]
    for class_id, class_name in isaac_classes.items():
        categories.append({
            'supercategory': class_name,
            'id': int(class_id),
            'name': class_name
        })
    
    # Initialize annotations structure
    annotations = {
        'images': [],
        'categories': categories,
        'annotations': []
    }
    
    image_id = 0
    annotation_id = 0
    
    # Get data directory
    data_dir = os.path.join(isaac_sim_path, split)
    if not os.path.exists(data_dir):
        print(f"Warning: {split} directory not found in {isaac_sim_path}")
        return
    
    # Get all scene directories
    scene_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    scene_dirs.sort()
    
    print(f"Processing {len(scene_dirs)} scenes for {split} split...")
    
    for i, scene_dir in enumerate(scene_dirs):
        scene_path = os.path.join(data_dir, scene_dir)
        if i % 100 == 0:  # Print progress every 100 scenes
            print(f"Processing scene {i+1}/{len(scene_dirs)}: {scene_dir}")
        
        # Load scene metadata
        scene_camera_path = os.path.join(scene_path, 'scene_camera.json')
        scene_gt_path = os.path.join(scene_path, 'scene_gt.json')
        
        if not os.path.exists(scene_camera_path) or not os.path.exists(scene_gt_path):
            print(f"Skipping {scene_dir}: missing metadata files")
            continue
            
        with open(scene_camera_path, 'r') as f:
            camera_data = json.load(f)
        
        with open(scene_gt_path, 'r') as f:
            gt_data = json.load(f)
        
        # Get RGB images
        rgb_dir = os.path.join(scene_path, 'rgb')
        if not os.path.exists(rgb_dir):
            print(f"Skipping {scene_dir}: no RGB directory")
            continue
            
        rgb_files = [f for f in os.listdir(rgb_dir) if f.endswith('.png')]
        rgb_files.sort()
        
        # Process each RGB image
        for rgb_file in rgb_files:
            # Extract camera/view index from filename (convert 000000.png -> 0)
            view_idx = str(int(rgb_file.split('.')[0]))
            
            # Skip if no camera data for this view
            if view_idx not in camera_data:
                continue
                
            # Skip if no GT data for this view
            if view_idx not in gt_data:
                continue
                
            # Get camera intrinsics
            cam_K = camera_data[view_idx]['cam_K']
            
            # Image info
            file_name = f"{split}/{scene_dir}/rgb/{rgb_file}"
            
            # Process objects in this view
            view_annotations = []
            
            # gt_data[view_idx] contains a list of objects for this view
            for obj_data in gt_data[view_idx]:
                obj_id = obj_data['obj_id']
                
                # Get object bounding box (if available)
                obj_bb = obj_data.get('obj_bb', None)
                if obj_bb is None:
                    # Skip objects without bounding boxes
                    continue
                
                # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
                bbox = [obj_bb[0], obj_bb[1], obj_bb[2] - obj_bb[0], obj_bb[3] - obj_bb[1]]
                
                # Skip invalid bounding boxes
                if bbox[2] <= 0 or bbox[3] <= 0:
                    continue
                
                # Get pose information
                cam_R_m2c = obj_data['cam_R_m2c']
                cam_t_m2c = obj_data['cam_t_m2c']
                
                # Convert translation from mm to m (Isaac Sim data appears to be in mm)
                position = [t / 1000.0 for t in cam_t_m2c]
                
                # Create annotation
                obj_annotation = {
                    'id': annotation_id,
                    'image_id': image_id,
                    'relative_pose': {
                        'position': position,
                        'rotation': cam_R_m2c
                    },
                    'bbox': bbox,
                    'area': bbox[2] * bbox[3],
                    'iscrowd': 0,
                    'category_id': obj_id
                }
                
                view_annotations.append(obj_annotation)
                annotation_id += 1
            
            # Only add image if it has annotations
            if view_annotations:
                # Add image info
                img_annotation = {
                    'file_name': file_name,
                    'id': image_id,
                    'width': 640,  # Adjust based on your image size
                    'height': 480,  # Adjust based on your image size
                    'intrinsics': cam_K,
                    'type': 'synthetic'  # or 'real' depending on your data
                }
                
                annotations['images'].append(img_annotation)
                annotations['annotations'].extend(view_annotations)
                image_id += 1
    
    # Save annotations
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f'{split}.json')
    
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Saved {len(annotations['images'])} images and {len(annotations['annotations'])} annotations to {output_file}")

def main():
    # Paths
    isaac_sim_path = '/data/gplong/force_map_project/w-poet/isaac_sim_poet_dataset'
    output_path = '/data/gplong/force_map_project/w-poet/poet/dataset_files/isaac_sim_annotations'
    
    # Convert each split
    for split in ['train', 'val']:
        print(f"\nConverting {split} split...")
        convert_isaac_sim_to_poet(isaac_sim_path, output_path, split)
    
    # Copy classes.json to the output directory
    import shutil
    classes_src = os.path.join(isaac_sim_path, 'annotations', 'classes.json')
    classes_dst = os.path.join(output_path, 'classes.json')
    shutil.copy2(classes_src, classes_dst)
    print(f"\nCopied classes.json to {classes_dst}")

if __name__ == '__main__':
    main()