# ------------------------------------------------------------------------
# PoET: Pose Estimation Transformer for Single-View, Multi-Object 6D Pose Estimation
# Copyright (c) 2022 Thomas Jantos (thomas.jantos@aau.at), University of Klagenfurt - Control of Networked Systems (CNS). All Rights Reserved.
# Licensed under the BSD-2-Clause-License with no commercial use [see LICENSE for details]
# ------------------------------------------------------------------------

import json
import os


annotation_set = "val"
test_set = "full"
train_set = "full"
val_set = "full"

if annotation_set == "test":
    if test_set == "full":
        # Annotate the full test set
        data_paths = ['test/']
        annotation_path = 'test.json'
        only_keyframes = False
        img_type = 'real'
elif annotation_set == "train":
    if train_set == "full":
        # Annotate the full test set
        data_paths = ['train/']
        annotation_path = 'train.json'
        only_keyframes = False
        img_type = 'real'
elif annotation_set == "val":
    if train_set == "full":
        # Annotate the full test set
        data_paths = ['val/']
        annotation_path = 'val.json'
        only_keyframes = False
        img_type = 'real'
else:
    print("The test set '{}' is not supported".format(test_set))
    exit()

base_path = '/media/sebastian/TEMP/poet/datasets/hope/'
output_base_path = '/media/sebastian/TEMP/poet/datasets/hope/annotations/'

categories = [
    {'supercategory': 'AlphabetSoup', 'id': 1, 'name': 'AlphabetSoup'},
    {'supercategory': 'BBQSauce', 'id': 2, 'name': 'BBQSauce'},
    {'supercategory': 'Butter', 'id': 3, 'name': 'Butter'},
    {'supercategory': 'Cherries', 'id': 4, 'name': 'Cherries'},
    {'supercategory': 'ChocolatePudding', 'id': 5, 'name': 'ChocolatePudding'},
    {'supercategory': 'Cookies', 'id': 6, 'name': 'Cookies'},
    {'supercategory': 'Corn', 'id': 7, 'name': 'Corn'},
    {'supercategory': 'CreamCheese', 'id': 8, 'name': 'CreamCheese'},
    {'supercategory': 'GranolaBars', 'id': 9, 'name': 'GranolaBars'},
    {'supercategory': 'GreenBeans', 'id': 10, 'name': 'GreenBeans'},
    {'supercategory': 'Ketchup', 'id': 11, 'name': 'Ketchup'},
    {'supercategory': 'Macaroni&Cheese', 'id': 12, 'name': 'Macaroni&Cheese'},
    {'supercategory': 'Mayo', 'id': 13, 'name': 'Mayo'},
    {'supercategory': 'Milk', 'id': 14, 'name': 'Milk'},
    {'supercategory': 'Mushrooms', 'id': 15, 'name': 'Mushrooms'},
    {'supercategory': 'Mustard', 'id': 16, 'name': 'Mustard'},
    {'supercategory': 'OrangeJuice', 'id': 17, 'name': 'OrangeJuice'},
    {'supercategory': 'Parmesan', 'id': 18, 'name': 'Parmesan'},
    {'supercategory': 'Peaches', 'id': 19, 'name': 'Peaches'},
    {'supercategory': 'Peas&Carrots', 'id': 20, 'name': 'Peas&Carrots'},
    {'supercategory': 'Pineapple', 'id': 21, 'name': 'Pineapple'},
    {'supercategory': 'Popcorn', 'id': 22, 'name': 'Popcorn'},
    {'supercategory': 'Raisins', 'id': 23, 'name': 'Raisins'},
    {'supercategory': 'SaladDressing', 'id': 24, 'name': 'SaladDressing'},
    {'supercategory': 'Spaghetti', 'id': 25, 'name': 'Spaghetti'},
    {'supercategory': 'TomatoSauce', 'id': 26, 'name': 'TomatoSauce'},
    {'supercategory': 'Tuna', 'id': 27, 'name': 'Tuna'},
    {'supercategory': 'Yogurt', 'id': 28, 'name': 'Yogurt'},
]
annotations = {'images': [],
               'categories': categories,
               'annotations': []}
image_id = 0
annotation_id = 0
annotations_removed = 0
for data_path in data_paths:
    print("Annotating: {}".format(data_path))
    # Get List of all subdirectories
    image_dirs = [d.name for d in os.scandir(base_path + data_path) if d.is_dir()]
    image_dirs.sort()

    for img_dir in image_dirs:
        print("Image Directory: {}".format(img_dir))
        img_dir_path = base_path + data_path + img_dir + '/'
        img_names = [img for img in os.listdir(img_dir_path + 'rgb/') if img[img.rfind('.'):] in ['.png', '.jpg']]
        img_names.sort()
        with open(img_dir_path + 'scene_gt_info.json', 'r') as f:
            bbox_annotations = json.load(f)
        with open(img_dir_path + 'scene_gt.json', 'r') as f:
            pose_annotations = json.load(f)
        with open(img_dir_path + 'scene_camera.json', 'r') as f:
            camera_annotations = json.load(f)
        # Check if annotation length is the same
        n_imgs = len(img_names)
        if len(bbox_annotations) != n_imgs:
            raise ValueError
        if len(pose_annotations) != n_imgs:
            raise ValueError
        if len(camera_annotations) != n_imgs:
            raise ValueError

        # Iterate over all images and annotations and create dict entries
        for img_name, b_k, p_k, c_k in zip(img_names, bbox_annotations, pose_annotations, camera_annotations):
            img_path = img_dir_path + 'rgb/' + img_name
            img_annotation_counter = 0
            file_name = data_path + img_dir + '/rgb/' + img_name
            bbox_data = bbox_annotations[b_k]
            pose_data = pose_annotations[p_k]
            camera_data = camera_annotations[c_k]

            for bbox, pose, in zip(bbox_data, pose_data):
                # If percentage of visible pixels is close to 0 --> skip
                # In the BOP challenge all bounding boxes are annotated regardless of whether the bounding boxes are visible or not
                if bbox['visib_fract'] < 0.05:
                    annotations_removed += 1
                    continue
                # Check if bbox starts / ends outside of image --> set to 0 or img boundary simply
                x1 = bbox['bbox_obj'][0]
                y1 = bbox['bbox_obj'][1]
                x2 = bbox['bbox_obj'][0] + bbox['bbox_obj'][2]
                y2 = bbox['bbox_obj'][1] + bbox['bbox_obj'][3]

                if x1 < 0:
                    # Adjust upper left and width
                    bbox['bbox_obj'][2] = bbox['bbox_obj'][2] + bbox['bbox_obj'][0]
                    bbox['bbox_obj'][0] = 0

                if y1 < 0:
                    # Adjust upper left and height
                    bbox['bbox_obj'][3] = bbox['bbox_obj'][3] + bbox['bbox_obj'][1]
                    bbox['bbox_obj'][1] = 0

                if x2 >= 640:
                    # Adjust width
                    bbox['bbox_obj'][2] = 640 - bbox['bbox_obj'][0] - 1

                if y2 >= 480:
                    # Adjust height
                    bbox['bbox_obj'][3] = 480 - bbox['bbox_obj'][1] - 1

                obj_annotation = {
                    'id': annotation_id,
                    'image_id': image_id,
                    'relative_pose': {
                        'position': [t / 1000.0 for t in pose['cam_t_m2c']],
                        'rotation': pose['cam_R_m2c']
                    },
                    'bbox': bbox['bbox_obj'],
                    'bbox_info': bbox,
                    'area': bbox['bbox_obj'][2] * bbox['bbox_obj'][3],
                    'iscrowd': 0,
                    'category_id': pose['obj_id']
                }
                annotations['annotations'].append(obj_annotation)
                img_annotation_counter += 1
                annotation_id += 1

            # Check if there are annotations for the image, otherwise skip
            if img_annotation_counter == 0:
                print("Image skipped! No annotations valid!")
                continue
            img_annotation = {
                'file_name': file_name,
                'id': image_id,
                'width': 640,
                'height': 480,
                'intrinsics': camera_data['cam_K'],
                'type': img_type
            }
            annotations['images'].append(img_annotation)
            image_id += 1

print("Annotations Removed: {}".format(annotations_removed))
with open(output_base_path + annotation_path, 'w') as out_file:
    json.dump(annotations, out_file)

