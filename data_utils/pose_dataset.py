# ------------------------------------------------------------------------
# PoET: Pose Estimation Transformer for Single-View, Multi-Object 6D Pose Estimation
# Copyright (c) 2022 Thomas Jantos (thomas.jantos@aau.at), University of Klagenfurt - Control of Networked Systems (CNS). All Rights Reserved.
# Licensed under the BSD-2-Clause-License with no commercial use [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE_DEFORMABLE_DETR in the LICENSES folder for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Build a dataset for the pose estimation task. This includes loading the images and annotations consisting of
class, bounding box, relative pose and absolute poses. Moreover, data augmentation and bounding box pertubation is possible.
"""
import copy
import json
import logging
import os
from pathlib import Path

import torch
import torch.utils.data
import numpy as np
import random
from pycocotools import mask as coco_mask

try:
    from .torchvision_datasets import CocoDetection
    from util.misc import get_local_rank, get_local_size
    from util.quaternion_ops import quat2rot, rot2quat
    import data_utils.transforms as T
except ImportError:
    # 当作为独立脚本运行时的导入
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_utils.torchvision_datasets import CocoDetection
    from util.misc import get_local_rank, get_local_size
    from util.quaternion_ops import quat2rot, rot2quat
    import data_utils.transforms as T
from scipy.stats import truncnorm, uniform


class PoseDataset(CocoDetection):
    """
    Pose Estimation Dataset. Returns samples consisting of images and the target containing the class, bounding box and
    the pose.
    """
    def __init__(self, img_folder, ann_file, synthetic_background, transforms, return_masks, jitter=False,
                 jitter_probability=0.5, std=0.02, cache_mode=False, local_rank=0, local_size=1):
        """
        Args:
            img_folder (string): path to the directory containing the images
            ann_file (string): path to the file containing the annotations
            synthetic_background (string): path to the directory containing the background images for synthetic images
            transforms (callable): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
            return_masks (bool): Whether to include the segmentation mask
            jitter (bool): Apply jitter to the bounding box
            jitter_probability (float): Probability with which jitter is applied to the bounding box
            std (float): standard deviation of the jitter.
        """
        super(PoseDataset, self).__init__(img_folder, ann_file, synthetic_background,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ProcessPoseData(return_masks)
        self.jitter = jitter
        self.jitter_probability = jitter_probability
        self.std = std

    def __getitem__(self, idx):
        img, target = super(PoseDataset, self).__getitem__(idx)
        image_id = self.ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]
        
        # Load camera intrinsics from scene_camera.json if not in img_info
        camera_intrinsics = img_info.get('camera_intrinsics')
        if camera_intrinsics is None:
            camera_intrinsics = self._load_camera_intrinsics(img_info)
            
        # Skip invalid cameras (those not found in scene_camera.json)
        if camera_intrinsics is None:
            # Try next sample by recursively calling __getitem__ with next index
            next_idx = (idx + 1) % len(self.ids)
            if next_idx != idx:  # Avoid infinite loop
                return self.__getitem__(next_idx)
            else:
                # If all samples are invalid, return empty sample
                empty_target = {
                    'boxes': torch.zeros((0, 4), dtype=torch.float32),
                    'labels': torch.zeros((0,), dtype=torch.int64),
                    'image_id': torch.tensor([image_id]),
                    'area': torch.zeros((0,), dtype=torch.float32),
                    'iscrowd': torch.zeros((0,), dtype=torch.int64),
                    'orig_size': torch.as_tensor([int(img.size[1]), int(img.size[0])]),
                    'size': torch.as_tensor([int(img.size[1]), int(img.size[0])]),
                    'camera_intrinsics': None
                }
                return img, empty_target
            
        target = {'image_id': image_id, 'annotations': target, 'camera_intrinsics': camera_intrinsics}
        
        # Load force data from original scene files BEFORE processing
        target = self._load_force_data(img_info, target)
        
        img, target = self.prepare(img, target) # Transforms bbox from xywh to xyxy
        if self._transforms is not None:
            img, target = self._transforms(img, target)  # Transforms bbox from un-normalized xyxy to normalized cxcywh (transforms.py 341)

        if self.jitter:
            # For the bounding box center we sample from a truncated normal distribution limited by the bounding box
            # width and height for x and y respectively. For the width and height jitter we assume a maximal error of
            # 10% and sample from this error range uniformly.
            jitter_boxes = copy.deepcopy(target["boxes"])
            for box in jitter_boxes:
                # Apply bounding box jitter only with probability
                if random.random() < self.jitter_probability:
                    cxa, cxb = -box[2] / (2 * self.std), box[2] / (2 * self.std)
                    cya, cyb = -box[3] / (2 * self.std), box[3] / (2 * self.std)
                    wa, wb = -0.3 / self.std, 0.3 / self.std
                    ha, hb = -0.3 / self.std, 0.3 / self.std

                    box[0] = truncnorm.rvs(cxa, cxb, loc=box[0], scale=self.std)
                    box[1] = truncnorm.rvs(cya, cyb, loc=box[1], scale=self.std)
                    box[2] = box[2] * (1 + truncnorm.rvs(wa, wb, loc=0, scale=self.std))
                    box[3] = box[3] * (1 + truncnorm.rvs(ha, hb, loc=0, scale=self.std))

            target["jitter_boxes"] = jitter_boxes

        return img, target

    def _load_camera_intrinsics(self, img_info):
        """
        Load camera intrinsics from scene_camera.json for isaac_sim_poet_dataset_force_point
        Returns None if camera is invalid (not found in scene_camera.json)
        """
        try:
            # Extract scene_id and camera_id from file_name
            file_name = img_info['file_name']  # e.g., "train/000001/rgb/000000.png"
            parts = file_name.split('/')
            if len(parts) >= 4:
                set_name = parts[0]  # train/val
                scene_name = parts[1]  # 000001
                camera_name = parts[3].split('.')[0]  # 000000 (remove .png)
                
                # Convert camera name to the format used in scene_camera.json (remove leading zeros)
                camera_id = str(int(camera_name))  # "000000" -> "0"
                
                # Construct path to scene_camera.json
                scene_camera_path = os.path.join(self.root, set_name, scene_name, 'scene_camera.json')
                
                if os.path.exists(scene_camera_path):
                    with open(scene_camera_path, 'r') as f:
                        camera_data = json.load(f)
                    
                    # Get camera intrinsics for specific camera
                    if camera_id in camera_data:
                        cam_K = camera_data[camera_id].get('cam_K')
                        if cam_K:
                            return cam_K
                    else:
                        # Camera not found in scene_camera.json, this camera is invalid
                        print(f"[INFO] Camera {camera_id} not found in scene_camera.json for scene {scene_name}, skipping invalid camera")
                        return None
                            
        except Exception as e:
            print(f"[WARNING] Failed to load camera intrinsics: {e}")
        
        # Return None for invalid cameras instead of default intrinsics
        return None

    def _load_force_data(self, img_info, target):
        """
        Load force data from isaac_sim_poet_dataset_force_point scene_gt.json files
        """
        try:
            # Extract scene_id and camera_id from file_name
            file_name = img_info['file_name']  # e.g., "train/000001/rgb/000000.png"
            parts = file_name.split('/')
            if len(parts) >= 4:
                set_name = parts[0]  # train/val
                scene_name = parts[1]  # 000001
                camera_name = parts[3].split('.')[0]  # 000000 (remove .png)
                
                # Convert camera name to the format used in scene_gt.json (remove leading zeros)
                camera_id = str(int(camera_name))  # "000000" -> "0"
                
                # Construct path to scene_gt.json
                scene_gt_path = os.path.join(self.root, set_name, scene_name, 'scene_gt.json')
                
                if os.path.exists(scene_gt_path):
                    with open(scene_gt_path, 'r') as f:
                        scene_data = json.load(f)
                    
                    # Get data for specific camera view and load force data
                    force_data = self._create_force_data_from_isaac_sim_scene_gt(scene_data, camera_id, target)
                    if force_data is not None:
                        target.update(force_data)
                        
        except Exception as e:
            logging.warning(f"Failed to load force data: {e}")
            import traceback
            traceback.print_exc()
        
        return target
    
    def _create_force_data_from_isaac_sim_scene_gt(self, scene_data, camera_id, target):
        """
        Create force data from isaac_sim_poet_dataset_force_point scene_gt.json format
        
        Args:
            scene_data: Complete scene data from scene_gt.json
            camera_id: Camera ID string
            target: Current target dict with annotation information
            
        Returns:
            dict: Force-related data to add to target
        """
        # Get object information from target.annotations if available
        annotations = target.get('annotations', [])
        if not annotations:
            return None
            
        # Filter out objects with iscrowd flag
        valid_annotations = [obj for obj in annotations if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        num_objects = len(valid_annotations)
        if num_objects == 0:
            return None
        
        # Get camera view data - in new format it's directly a list under camera_id key
        view_data = scene_data.get(camera_id, [])
        if not view_data:
            return None
        
        # Get forces data - in new format it's stored at the top level
        forces_data = scene_data.get('forces', [])
        
        # Create mapping from obj_id to index in target arrays
        # We need to match objects from annotation with objects in scene_gt
        obj_id_to_target_idx = {}
        
        # Build the mapping using object order since COCO annotation preserves object order
        for target_idx, obj_data in enumerate(view_data):
            obj_id = obj_data.get('obj_id')
            if obj_id is not None:
                obj_id_to_target_idx[obj_id] = target_idx
        
        # Initialize force-related data structures
        force_matrix = torch.zeros((num_objects, num_objects, 3), dtype=torch.float32)
        contact_forces = torch.zeros((num_objects, 3), dtype=torch.float32) 
        contact_positions = torch.zeros((num_objects, 3), dtype=torch.float32)
        friction_forces = torch.zeros((num_objects, 3), dtype=torch.float32)
        friction_positions = torch.zeros((num_objects, 3), dtype=torch.float32)
        environment_forces = torch.zeros((num_objects, 3), dtype=torch.float32)  # For table forces
        environment_contact_pos = torch.zeros((num_objects, 3), dtype=torch.float32)
        environment_friction_forces = torch.zeros((num_objects, 3), dtype=torch.float32)
        environment_friction_pos = torch.zeros((num_objects, 3), dtype=torch.float32)
        
        # Process force interactions
        force_count = 0
        for force_interaction in forces_data:
            source_obj_id = force_interaction.get('source_obj_id')
            target_obj_id = force_interaction.get('target_obj_id')
            
            # Get target object index
            if target_obj_id not in obj_id_to_target_idx:
                continue
                
            target_idx = obj_id_to_target_idx[target_obj_id]
            
            # Extract force components
            contact_force = force_interaction.get('contact_force', [0.0, 0.0, 0.0])
            contact_pos = force_interaction.get('contact_pos', [0.0, 0.0, 0.0])
            friction_force = force_interaction.get('friction_force', [0.0, 0.0, 0.0])
            friction_pos = force_interaction.get('friction_pos', [0.0, 0.0, 0.0])
            
            if source_obj_id == -1:
                # Environment force (e.g., table)
                environment_forces[target_idx] = torch.tensor(contact_force, dtype=torch.float32)
                environment_contact_pos[target_idx] = torch.tensor(contact_pos, dtype=torch.float32)
                environment_friction_forces[target_idx] = torch.tensor(friction_force, dtype=torch.float32)
                environment_friction_pos[target_idx] = torch.tensor(friction_pos, dtype=torch.float32)
                force_count += 1
            else:
                # Object-to-object force
                if source_obj_id in obj_id_to_target_idx:
                    source_idx = obj_id_to_target_idx[source_obj_id]
                    
                    # Store in force matrix
                    force_matrix[source_idx, target_idx, :] = torch.tensor(contact_force, dtype=torch.float32)
                    
                    # Also store individual force components for target object
                    contact_forces[target_idx] += torch.tensor(contact_force, dtype=torch.float32)
                    friction_forces[target_idx] += torch.tensor(friction_force, dtype=torch.float32)
                    
                    # Store positions (use last valid position if multiple forces on same object)
                    contact_positions[target_idx] = torch.tensor(contact_pos, dtype=torch.float32)
                    friction_positions[target_idx] = torch.tensor(friction_pos, dtype=torch.float32)
                    
                    force_count += 1
        
        if force_count > 0:
            return {
                'force_matrix': force_matrix,
                'contact_forces': contact_forces,
                'contact_positions': contact_positions, 
                'friction_forces': friction_forces,
                'friction_positions': friction_positions,
                'environment_forces': environment_forces,
                'environment_contact_pos': environment_contact_pos,
                'environment_friction_forces': environment_friction_forces,
                'environment_friction_pos': environment_friction_pos
            }
        else:
            return None


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ProcessPoseData(object):
    """
    Processes the annotation file and brings it in the right format for the pose estimation task.
    """
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        
        # Extract force data from top-level target if available
        input_force_data = {}
        force_keys = ['contact_forces', 'contact_positions', 'friction_forces', 'friction_positions',
                     'environment_forces', 'environment_contact_pos', 'environment_friction_forces',
                     'environment_friction_pos', 'force_matrix']
        for key in force_keys:
            if key in target:
                input_force_data[key] = target[key]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # Transform xywh to xyxy
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        masks = None
        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        # Load absolute camera pose
        # Only need to store the global camera pose from the first annotated object as it is the same for each object
        cam_position = None
        cam_rotation = None
        # TODO: Implement if rotation stored as quaternions
        if 'camera_pose' in anno[0]:
            if 'position' in anno[0]['camera_pose']:
                cam_position = anno[0]['camera_pose']['position']
                cam_position = torch.tensor(cam_position, dtype=torch.float32)
            if 'rotation' in anno[0]['camera_pose']:
                cam_rotation = anno[0]['camera_pose']['rotation']
                cam_rotation = torch.tensor(cam_rotation, dtype=torch.float32)
                cam_rotation = torch.reshape(cam_rotation, (3, 3))

        # Load absolute object pose
        obj_position = None
        obj_rotation = None
        if 'object_pose' in anno[0]:
            if 'position' in anno[0]['object_pose']:
                obj_position = [obj['object_pose']['position'] for obj in anno]
                obj_position = torch.tensor(obj_position, dtype=torch.float32)
            if 'rotation' in anno[0]['object_pose']:
                obj_rotation = [obj['object_pose']['rotation'] for obj in anno]
                obj_rotation = torch.tensor(obj_rotation, dtype=torch.float32)
                obj_rotation = torch.reshape(obj_rotation, (-1, 3, 3))

        # Load relative pose between camera and object
        rel_position = None
        rel_quaternion = None
        rel_rotation = None
        if 'relative_pose' in anno[0]:
            if 'position' in anno[0]['relative_pose']:
                rel_position = [obj["relative_pose"]['position'] for obj in anno]
                rel_position = torch.tensor(rel_position, dtype=torch.float32)
            if 'quaternions' in anno[0]['relative_pose']:
                rel_quaternion = [obj["relative_pose"]['quaternions'] for obj in anno]
                rel_quaternion = torch.tensor(rel_quaternion, dtype=torch.float32)
            if 'rotation' in anno[0]['relative_pose']:
                rel_rotation = [obj["relative_pose"]['rotation'] for obj in anno]
                rel_rotation = torch.tensor(rel_rotation, dtype=torch.float32)
                if rel_rotation.shape[1] == 9:
                    rel_rotation = torch.reshape(rel_rotation, (-1, 3, 3))
                rel_quaternion = rot2quat(rel_rotation)
                rel_quaternion = torch.tensor(rel_quaternion, dtype=torch.float32)
            else:
                q = np.array([obj["relative_pose"]['quaternions'] for obj in anno])
                rel_rotation = quat2rot(q)
                rel_rotation = torch.tensor(rel_rotation, dtype=torch.float32)
        else:
            # If no relative_pose field, use object_pose as relative pose
            # Based on analysis, object_pose appears to be already in camera coordinate system
            if obj_position is not None:
                rel_position = obj_position.clone()
            if obj_rotation is not None:
                rel_rotation = obj_rotation.clone()
                rel_quaternion = rot2quat(rel_rotation.numpy())
                rel_quaternion = torch.tensor(rel_quaternion, dtype=torch.float32)

        intrinsics = None
        if 'camera_intrinsics' in target and target['camera_intrinsics']:
            cam_intrinsics = torch.as_tensor(target['camera_intrinsics'], dtype=torch.float32)
            intrinsics = cam_intrinsics.unsqueeze(0).repeat(len(anno), 1, 1)
        elif anno and 'intrinsics' in anno[0]:
            intrinsics = [obj['intrinsics'] for obj in anno]
            intrinsics = torch.as_tensor(intrinsics, dtype=torch.float32)
        # No default intrinsics for invalid cameras - they should be skipped at dataset level
        
        # Load mass and force information
        masses = None
        force_matrix = None
        if anno and 'mass' in anno[0]:
            masses = [obj.get('mass', 0.0) for obj in anno]
            masses = torch.tensor(masses, dtype=torch.float32)
        
        # Load forces if available from annotation (legacy support)
        if anno and 'forces' in anno[0]:
            # Create force matrix N x N x 3
            num_objects = len(anno)
            force_matrix = torch.zeros((num_objects, num_objects, 3), dtype=torch.float32)
            
            # Build obj_id to index mapping for this image
            obj_ids = [obj.get('obj_id', i) for i, obj in enumerate(anno)]
            id_to_idx = {obj_id: idx for idx, obj_id in enumerate(obj_ids)}
            
            # Fill force matrix from forces data
            force_count = 0
            for obj_idx, obj in enumerate(anno):
                forces_list = obj.get('forces', [])
                
                if forces_list:  # If this object has force interactions
                    for force_data in forces_list:
                        source_obj_id = force_data.get('source_obj_id')
                        target_obj_id = force_data.get('target_obj_id')
                        force_vector = force_data.get('force_vector', [0.0, 0.0, 0.0])
                        
                        # Map obj_ids to array indices
                        if source_obj_id in id_to_idx and target_obj_id in id_to_idx:
                            source_idx = id_to_idx[source_obj_id]
                            target_idx = id_to_idx[target_obj_id]
                            force_matrix[source_idx, target_idx, :] = torch.tensor(force_vector, dtype=torch.float32)
                            force_count += 1

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks and masks is not None:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]
        if obj_position is not None:
            obj_position = obj_position[keep]
        if obj_rotation is not None:
            obj_rotation = obj_rotation[keep]
        if rel_position is not None:
            rel_position = rel_position[keep]
        if rel_quaternion is not None:
            rel_quaternion = rel_quaternion[keep]
        if rel_rotation is not None:
            rel_rotation = rel_rotation[keep]
        if masses is not None:
            masses = masses[keep]
        if force_matrix is not None:
            # Apply keep mask to force matrix (both dimensions)
            force_matrix = force_matrix[keep][:, keep]
            
        # Apply keep mask to force-related data from isaac_sim_poet_dataset_force_point
        contact_forces = input_force_data.get('contact_forces')
        contact_positions = input_force_data.get('contact_positions')
        friction_forces = input_force_data.get('friction_forces')
        friction_positions = input_force_data.get('friction_positions')
        environment_forces = input_force_data.get('environment_forces')
        environment_contact_pos = input_force_data.get('environment_contact_pos')
        environment_friction_forces = input_force_data.get('environment_friction_forces')
        environment_friction_pos = input_force_data.get('environment_friction_pos')
        force_matrix_isaac = input_force_data.get('force_matrix')
        
        if contact_forces is not None:
            contact_forces = contact_forces[keep]
        if contact_positions is not None:
            contact_positions = contact_positions[keep]
        if friction_forces is not None:
            friction_forces = friction_forces[keep]
        if friction_positions is not None:
            friction_positions = friction_positions[keep]
        if environment_forces is not None:
            environment_forces = environment_forces[keep]
        if environment_contact_pos is not None:
            environment_contact_pos = environment_contact_pos[keep]
        if environment_friction_forces is not None:
            environment_friction_forces = environment_friction_forces[keep]
        if environment_friction_pos is not None:
            environment_friction_pos = environment_friction_pos[keep]
        if force_matrix_isaac is not None:
            force_matrix_isaac = force_matrix_isaac[keep][:, keep]
            
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks and masks is not None:
            target["masks"] = masks
        target["image_id"] = image_id

        if keypoints is not None:
            target["keypoints"] = keypoints
        if cam_position is not None:
            target["camera_position_w"] = cam_position
        if cam_rotation is not None:
            target["camera_rotation_w"] = cam_rotation
        if obj_position is not None:
            target["object_position_w"] = obj_position
        if obj_rotation is not None:
            target["object_rotation_w"] = obj_rotation
        if rel_position is not None:
            target["relative_position"] = rel_position
        if rel_quaternion is not None:
            target["relative_quaternions"] = rel_quaternion
        if rel_rotation is not None:
            target["relative_rotation"] = rel_rotation
        if intrinsics is not None:
            target["intrinsics"] = intrinsics
        if masses is not None:
            target["masses"] = masses
        if force_matrix is not None:
            target["force_matrix"] = force_matrix
            
        # Add isaac_sim_poet_dataset_force_point specific force data
        if contact_forces is not None:
            target["contact_forces"] = contact_forces
        if contact_positions is not None:
            target["contact_positions"] = contact_positions
        if friction_forces is not None:
            target["friction_forces"] = friction_forces
        if friction_positions is not None:
            target["friction_positions"] = friction_positions
        if environment_forces is not None:
            target["environment_forces"] = environment_forces
        if environment_contact_pos is not None:
            target["environment_contact_pos"] = environment_contact_pos
        if environment_friction_forces is not None:
            target["environment_friction_forces"] = environment_friction_forces
        if environment_friction_pos is not None:
            target["environment_friction_pos"] = environment_friction_pos
        if force_matrix_isaac is not None:
            target["force_matrix"] = force_matrix_isaac  # Override with Isaac Sim force matrix if available

        # Expand force matrix to include environment forces (N+1 x N+1)
        final_force_matrix = target.get("force_matrix")
        if final_force_matrix is not None and environment_forces is not None:
            num_objects = final_force_matrix.shape[0]
            # Create expanded force matrix (N+1) x (N+1) x 3
            expanded_force_matrix = torch.zeros((num_objects + 1, num_objects + 1, 3), dtype=torch.float32)
            
            # Copy original object-to-object forces (upper-left N x N block)
            expanded_force_matrix[:num_objects, :num_objects] = final_force_matrix
            
            # Fill environment forces: environment -> objects (last column, excluding last element)
            expanded_force_matrix[:num_objects, num_objects] = environment_forces
            
            # Fill reaction forces: objects -> environment (last row, excluding last element)
            # According to Newton's third law: F_reaction = -F_action
            expanded_force_matrix[num_objects, :num_objects] = -environment_forces
            
            # Environment self-interaction (last element) remains zero
            expanded_force_matrix[num_objects, num_objects] = torch.zeros(3, dtype=torch.float32)
            
            target["force_matrix"] = expanded_force_matrix
        else:
            print("[WARNING] Force matrix not found or environment forces not available, skipping expansion.")
            print("final_force_matrix:", final_force_matrix is not None, "environment_forces:", environment_forces is not None)
        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_pose_estimation_transform(image_set, use_rgb_augmentation=False, use_grayscale=False):
    """
    Apply transformations to the images and targets for the pose estimation task depending on the data split.
    """
    # TODO: Add proper data augmentation for pose estimation

    if use_grayscale and image_set not in ['keyframes', 'keyframes_bop', 'test']:
        normalize = T.Compose([
            T.GrayScale(),
            T.ToTensor(),
            T.To3DImage(),
            T.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        ])
    else:
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        ])

    rgb_augmentation = T.Compose([T.Blur(),
                                  T.Sharpness(),
                                  T.Contrast(),
                                  T.Brightness(),
                                  T.Color()])

    if image_set == 'train':
        if use_rgb_augmentation:
            return T.Compose([rgb_augmentation, normalize, ])
        else:
            return T.Compose([normalize, ])

    if image_set == 'train_synt':
        if use_rgb_augmentation:
            return T.Compose([rgb_augmentation, normalize, ])
        else:
            return T.Compose([normalize, ])

    if image_set == 'train_pbr':
        if use_rgb_augmentation:
            return T.Compose([rgb_augmentation, normalize, ])
        else:
            return T.Compose([normalize, ])

    if image_set == 'val':
        return T.Compose([
            normalize,
        ])

    if image_set == 'test' or "test_all":
        return T.Compose([
            normalize,
        ])

    if image_set in ['keyframes', 'keyframes_bop']:
        return T.Compose([
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.dataset_path)
    assert root.exists(), f'provided dataset path {root} does not exist'
    PATHS = {
        "train": (root, root / "annotations" / f'train.json'),
        "train_synt": (root, root / "annotations" / f'train_synt.json'),
        "train_pbr": (root, root / "annotations" / f'train_pbr.json'),
        "test": (root, root / "annotations" / f'test.json'),
        "test_all": (root, root / "annotations" / f'test_all.json'),
        "keyframes": (root, root / "annotations" / f'keyframes.json'),
        "keyframes_bop": (root, root / "annotations"/ f'keyframes_bop.json'),
        "val": (root, root / "annotations" / f'val.json'),
    }

    img_folder, ann_file = PATHS[image_set]

    # TODO: Replace 'transforms' by a proper data augmentation function suitable for pose estimation. Currently only
    #  image level augmentation possible (e.g. color augmentation, noise).
    if args.bbox_mode == 'jitter':
        jitter = True
    else:
        jitter = False
    dataset = PoseDataset(img_folder, ann_file, synthetic_background=args.synt_background,
                          transforms=make_pose_estimation_transform(image_set, args.rgb_augmentation, args.grayscale),
                          return_masks=False, jitter=jitter, jitter_probability=args.jitter_probability,
                          cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    return dataset
