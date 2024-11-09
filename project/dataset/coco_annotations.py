"""
This script is used to transform a recorded custom dataset to the COCO annotation format to be used in PoET.
"""

import os
import json
from common import categories, dimensions


def file_exists_in_directory(directory: str, filename: str) -> bool:
    return os.path.isfile(os.path.join(directory, filename))

RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
RESET = '\033[97m'

dataset = "val"
base_path = "/media/sebastian/TEMP/poet/datasets/custom_doll"
main_dir = f"{base_path}/{dataset}"

annotations_file = f"{base_path}/annotations/{dataset}.json"

# Initialize COCO format dictionary
coco_annotations = {
    'images': [],
    'categories': categories,
    'annotations': []
}

image_id = 0
annotation_id = 0

# Iterate through each subdirectory in the main directory
for subdirectory in os.listdir(main_dir):
    print("[INFO] Processing " + subdirectory + " ..")
    sub_dir_path = os.path.join(main_dir, subdirectory)
    if os.path.isdir(sub_dir_path):
        gt_obj_path = os.path.join(sub_dir_path, f"gt_obj.json")
        
        # Check if the corresponding text file exists
        if not os.path.exists(gt_obj_path):
            print(f"{RED}[FAIL]{RESET} Cannot find {gt_obj_path}! Exit program ...")
            exit()

        with open(gt_obj_path, 'r') as file:
            content = json.load(file)
            for line in content:
                # Extract filename, translation, and rotation
                filename = line + ".png"

                rgb_path = os.path.join(sub_dir_path, "rgb/")
                if not file_exists_in_directory(rgb_path, filename):
                    print(f"{RED}[FAIL]{RESET} Cannot find {os.path.join(rgb_path, filename)}! Exit program ...")
                    exit(-1)

                translation = list(content[line]['0']['t'])
                rotation = list(content[line]['0']['rot'])

                rotation_matrix_flat = [i for row in rotation for i in row]
                relative_pose = {
                    "position": translation,
                    "rotation": rotation_matrix_flat
                }

                # Image information
                image_info = {
                    "file_name": os.path.join(dataset, subdirectory, "rgb", filename),
                    "id": image_id,
                    "width": 640,
                    "height": 480,
                    "intrinsics": [],
                    "type": "real"
                }
                coco_annotations['images'].append(image_info)

                bbox = content[line]['0']['box']
                x, y, w, h = bbox
                class_id = content[line]['0']['class']

                # Bounding box information
                bbox_info = {
                    "bbox_obj": bbox,
                    "bbox_visib": bbox,
                    "px_count_all": w * h,
                    "px_count_valid": 0,
                    "px_count_visib": 0,
                    "visib_fract": 1.0
                }

                # Annotation information
                # !!! COCO requires bbox in xywh format !!!
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "relative_pose": relative_pose,
                    "bbox": bbox,
                    "bbox_info": bbox_info,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "category_id": class_id
                }
                coco_annotations['annotations'].append(annotation)
                
                annotation_id += 1
                image_id += 1

            print(f"{GREEN}[SUCC]{RESET} Processed {subdirectory} ..\n")


# Write the COCO annotations to a file
with open(annotations_file, 'w') as f:
    json.dump(coco_annotations, f, indent=4)

print("COCO annotations file created successfully.")
