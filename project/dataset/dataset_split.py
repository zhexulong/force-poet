import os
import json
import re
import shutil
import random
from pathlib import Path

TRAIN_PERCENT = 0.80
TEST_PERCENT = 0.15
SEED = 42

# Paths
# BASE = "."
BASE = "../dataset_doll"
base_path = f"{BASE}/records"  # Path to the main folder containing the object recordings
train_path = f"{BASE}/train"
val_path = f"{BASE}/val"
test_path = f"{BASE}/test"

# Create train/test directories if they don't exist
os.makedirs(train_path, exist_ok=False)
os.makedirs(val_path, exist_ok=False)
os.makedirs(test_path, exist_ok=False)

# Initialize counters
total_train_images = 0
total_test_images = 0
total_val_images = 0
total_folders = 0


def split_data(data: dict, train_percent: float = 0.8, test_percent: float = 0.2) -> (list, list, list):
    """
    Split data into training, validation and test sets.
    """
    frames = list(data.keys())
    random.seed(SEED)
    random.shuffle(frames)

    # Train / Val split
    split_idx = int(len(frames) * train_percent)
    train_frames = frames[:split_idx]
    val_frames = frames[split_idx:]

    random.shuffle(val_frames)
    # Test split
    split_idx = int(len(val_frames) * test_percent)

    # If split_idx is 0 and there are more than 2 validation images,
    # select at least one test image.
    if split_idx < 1 and len(val_frames) > 2:
        split_idx = 1

    val_frames = val_frames[split_idx:]
    test_frames = val_frames[:split_idx]
    return train_frames, val_frames, test_frames


def extract_frame_number(frame_name: str) -> int:
    """
    Extract frame number from frame name (e.g. "Frame123" -> 123).
    """
    return int(re.search(r'\d+', frame_name).group())


def save_split_data(data: dict, frames: list, save_path: str, file_name: str) -> None:
    """
    Save split meta-data to disk.
    """
    split_data = {frame: data[frame] for frame in frames}
    # Sort frames based on the extracted numeric part of the frame name in ascending order
    sorted_data = dict(sorted(split_data.items(), key=lambda x: extract_frame_number(x[0])))
    # del sorted_data["reverse"]
    with open(os.path.join(save_path, file_name), 'w') as f:
        json.dump(sorted_data, f, indent=4)


# Iterate through each object folder
for obj_folder in os.listdir(base_path):
    obj_path = os.path.join(base_path, obj_folder)
    if os.path.isdir(obj_path):
        cam_gt_file = os.path.join(obj_path, "gt_cam.json")
        obj_gt_file = os.path.join(obj_path, "gt_obj.json")
        rgb_folder = os.path.join(obj_path, "rgb")
        
        # Skip if the required files do not exist
        if not (os.path.exists(cam_gt_file) and os.path.exists(obj_gt_file) and os.path.isdir(rgb_folder)):
            print(f"[ERR] {cam_gt_file} or {obj_gt_file} or {rgb_folder} is missing!")
            continue

        # Load the ground truth JSON files
        with open(cam_gt_file, 'r') as f:
            cam_gt_data = json.load(f)
        
        with open(obj_gt_file, 'r') as f:
            obj_gt_data = json.load(f)

        # Split the frames into train and test sets
        train_frames, val_frames, test_frames = split_data(cam_gt_data, TRAIN_PERCENT, TEST_PERCENT)

        # Update object counter
        total_folders += 1

        # Create directories for this object's train and test data
        obj_train_path = os.path.join(train_path, obj_folder)
        obj_val_path = os.path.join(val_path, obj_folder)
        obj_test_path = os.path.join(test_path, obj_folder)
        os.makedirs(os.path.join(obj_train_path, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(obj_val_path, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(obj_test_path, "rgb"), exist_ok=True)

        # Save training data
        save_split_data(cam_gt_data, train_frames, obj_train_path, "gt_cam.json")
        save_split_data(obj_gt_data, train_frames, obj_train_path, "gt_obj.json")

        # Save validation data
        save_split_data(cam_gt_data, val_frames, obj_val_path, "gt_cam.json")
        save_split_data(obj_gt_data, val_frames, obj_val_path, "gt_obj.json")

        # Save testing data
        save_split_data(cam_gt_data, test_frames, obj_test_path, "gt_cam.json")
        save_split_data(obj_gt_data, test_frames, obj_test_path, "gt_obj.json")

        # Copy corresponding images to train/val/test folders and update counters
        for frame in train_frames:
            img_name = f"{frame}.png"  # Assuming images are named based on frame numbers
            img_src = os.path.join(rgb_folder, img_name)
            if os.path.exists(img_src):
                shutil.copy2(img_src, os.path.join(obj_train_path, "rgb", img_name))
                total_train_images += 1

        for frame in val_frames:
            img_name = f"{frame}.png"
            img_src = os.path.join(rgb_folder, img_name)
            if os.path.exists(img_src):
                shutil.copy2(img_src, os.path.join(obj_val_path, "rgb", img_name))
                total_val_images += 1

        for frame in test_frames:
            img_name = f"{frame}.png"
            img_src = os.path.join(rgb_folder, img_name)
            if os.path.exists(img_src):
                shutil.copy2(img_src, os.path.join(obj_test_path, "rgb", img_name))
                total_test_images += 1

# Print the final counts
print(f"Total Folders: {total_folders}")
print(f"Total training images: {total_train_images}")
print(f"Total validation images: {total_val_images}")
print(f"Total testing images: {total_test_images}")
print(f"Total images: {total_test_images + total_train_images + total_val_images}")