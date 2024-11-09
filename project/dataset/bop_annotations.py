import re
import cv2
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
import json
import numpy as np
import os
import shutil
from torchvision.ops import box_convert
from groundingdino.util.inference import load_model, load_image, predict, annotate
import torch

from common import categories, dimensions

model = load_model("/home/sebastian/repos/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "/home/sebastian/repos/GroundingDINO/weights/groundingdino_swint_ogc.pth")
# IMAGE_PATH = "input/chair2.png"
# TEXT_PROMPT = "object in the middle"
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
RESET = '\033[97m'

def is_image_present(folder_path, image_name):
    image_path = os.path.join(folder_path, image_name + ".png")
    return os.path.isfile(image_path)


def get_position(content):
    return [float(content[1]), float(content[2]), float(content[3])]


def get_rotation(content):
    return [float(content[4]), float(content[5]), float(content[6]), float(content[7])]

def get_category_name(category_id, categories):
    for category in categories:
        if category['id'] == category_id:
            return category['name']
    
    return 'object'


def get_category_id(name):
    """
    Returns the category ID based on the category name.

    :param categories: List of category dictionaries.
    :param name: The name of the category to search for.
    :return: The ID of the category if found, else None.
    """
    for category in categories:
        if category['name'] == name:
            return category['id']
    return None


def unnormalize_bboxes(boxes, img_width, img_height):
    """
    Unnormalizes bboxes in cxcywh format.
    """
    unnormalized_bboxes = torch.empty_like(boxes)
    unnormalized_bboxes[:, 0] = boxes[:, 0] * img_width   # cx
    unnormalized_bboxes[:, 1] = boxes[:, 1] * img_height  # cy
    unnormalized_bboxes[:, 2] = boxes[:, 2] * img_width   # w
    unnormalized_bboxes[:, 3] = boxes[:, 3] * img_height  # h
    return unnormalized_bboxes


def convert_to_pixel_values(img_width, img_height, bbox):
    """
    Convert normalized bbox to pixel values.
    """
    cx, cy, w, h = bbox
    x_min = int((cx - w / 2) * img_width)
    y_min = int((cy - h / 2) * img_height)
    width_ = int(w * img_width)
    height_ = int(h * img_height)
    return x_min, y_min, width_, height_


def visualize_bounding_box(image, bbox):
    """
    Visualizes given bbox in given image.
    """
    x, y, w, h = bbox
    cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)

    return image


def get_center_of_object(object: str):
    """
    Returns the center of object (in pixels) of the given object.
    """
    h = dimensions[object]["h"]
    return np.array([0, 0, h / 2])


def center_object_origin(object: str, t: np.ndarray):
    """
    Centers the given origin (t) of an object.
    """
    center = get_center_of_object(object)
    return t + center


def strip_numeric_suffix(s: str):
    """
    Strips numerical suffixes from string s (e.g. 'doll_1' -> 'doll').
    """
    return re.sub(r'_\d+$', '', s)


def correct_camera_tilt(rotation: np.ndarray, angle: float = 10.0):
    """
    Corrects the tilt of a given camera rotation.
    """
    rot = Rotation.from_matrix(rotation)
    rot_y = Rotation.from_euler('y', angle, degrees=True)
    return rot * rot_y


base_path = f"../dataset_doll/records"

width = 640
height = 480

cnt = 0
recordings = os.listdir(base_path)
for object in recordings:
    print(f"[INFO] Processing '{object}' ...")

    obj = strip_numeric_suffix(object)
    if obj not in dimensions:
        print(f"{RED}[ERR ]{RESET} Cannot find '{obj}' in 'dimensions'! Skipping...")
        print("--------------------------------------------------")
        continue

    # if os.path.isfile(f"{base_path}/{object}/gt_cam.json"):
    #     print(f"{RED}[FAIL]{RESET} gt_cam.json already exists for {object}! Skipping...")
    #     print("--------------------------------------------------")
    #     continue
    #
    # if os.path.isfile(f"{base_path}/{object}/gt_obj.json"):
    #     print(f"{RED}[FAIL]{RESET} gt_obj.json already exists for {object}! Skipping...")
    #     print("--------------------------------------------------")
    #     continue

    if os.path.exists(f"{base_path}/{object}/rgb/"):
        print(f"{YELLOW}[WARN]{RESET} rgb/ folder already exists for {object}!")
        shutil.rmtree(f"{base_path}/{object}/rgb/")
        os.makedirs(f"{base_path}/{object}/rgb/")
        # print(f"{RED}[FAIL]{RESET} rgb/ folder already exists for {object}! Skipping...")
        # print("--------------------------------------------------")
        # continue
    else:
        os.makedirs(f"{base_path}/{object}/rgb/")

    bbox_dir = f"{base_path}/{object}/bbox/"
    if os.path.exists(bbox_dir):
        print(f"{YELLOW}[WARN]{RESET} bbox/ folder already exists for {object}!")
        shutil.rmtree(f"{base_path}/{object}/bbox/")
        os.makedirs(bbox_dir)
        # print(f"{RED}[FAIL]{RESET} bbox/ folder already exists for {object}! Skipping...")
        # print("--------------------------------------------------")
        # continue
    else:
        os.makedirs(bbox_dir)
        print(f"[INFO] Created 'bbox' directory '{bbox_dir}'")

    # This is the raw file of the Drone's ground truth pose recorded the optitrack system.
    with open(f"{base_path}/{object}/gt_cam.txt") as file:
        lines = [line.rstrip() for line in file]

    idx = 0
    data_cam = {}
    data_obj = {}
    for line in lines:
        # if (idx % 100) != 0: 
        #     idx += 1
        #     continue
        # idx += 1

        content = line.split(" ")
        name = ""
        if '/' in content[0]:
            name = content[0].split("/")[1].split(".")[0]
        else:
            name = content[0].split(".")[0]

        position = np.array(get_position(content))
        rotation = Rotation.from_quat(get_rotation(content)).as_matrix()

        # rotation = correct_camera_tilt(rotation).as_matrix()

        if not is_image_present(f"{base_path}/{object}/imgs", name):
            print(f"[WARN] Cannot find {base_path}/{object}/imgs/{name}!")
            continue

        # R,G,B
        image_source, image = load_image(os.path.join(f"{base_path}/{object}", f"imgs/{name}.png"))
        image_source = cv2.resize(image_source, (width, height))

        tmp = obj
        if "_" in obj:
            split = tmp.split("_")
            tmp = split[0]
            for i, o in enumerate(split):
                if i != 0 and not o.isnumeric():
                    tmp = tmp + " " + o

        ## GroundingDINO prompt
        if (tmp == 'doll'):
            prompt = "human"
        prompt = tmp + "."
        # prompt = get_category_name(int(subdirectory), categories=categories)
        # prompt = "object in the middle."
        # prompt = "chair"

        # bbox format: normalized (!) cxcywh
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=prompt,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )

        if len(boxes) == 0:
            print(f"{RED}[ERR ]{RESET} Could not detect bbox for {name} in {object}! Skipping image...")
            continue

        if len(boxes) != len(phrases):
            print(f"{YELLOW}[WARN]{RESET} The number of boxes and phrases for {name} in {object} isn't equal?! Skipping image...")
            continue

        # !!! COCO (and PoET) requires bbox in (unnormalized) xywh format !!!
        boxes = unnormalize_bboxes(boxes, width, height)
        boxes_xywh = box_convert(boxes, "cxcywh", "xywh")

        ## TODO: REFACTORE!
        # Select bbox with largest area
        bbox = None
        if len(boxes_xywh) > 1:
            print(f"{YELLOW}[WARN]{RESET} Detected more then one bbox for {name} in {object}! Selecting the one with the largest area ..")

            idx = 0
            area = 0
            for i, box in enumerate(boxes_xywh.numpy()):
                tmp = box[2] * box[3]
                if area < tmp:
                    area = tmp
                    idx = i
            bbox = boxes_xywh[idx]
        else:
            bbox = boxes_xywh[0]

        if bbox == None:
            print(f"{RED}[ERR ]{RESET} Did not select a bounding box for {name} in {object}!?")

        ################################
        # BBox Visualization
        #

        image_cv = np.copy(image_source)
        image_with_boxes = visualize_bounding_box(image_cv, bbox)

        x, y, w, h = bbox  # xywh format (x, y is the top-left corner, w is width, h is height)
        phrase = phrases[0]  # Corresponding predicted phrase

        # Define position for the text (above the top-left corner of the box)
        text_position = (int(x), int(y) - 10)

        # Draw text on the image
        cv2.putText(image_with_boxes, phrase, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        plt.figure(figsize=(6, 6))
        plt.imshow(image_with_boxes)
        plt.axis('off')

        bbox_file = os.path.join(bbox_dir, name + ".png")
        plt.savefig(bbox_file, bbox_inches="tight")
        plt.close()

        # plt.show()

        bbox = bbox.tolist()

        class_id = get_category_id(obj)
        if class_id is None:
            print(f"{RED}[ERR ]{RESET} Cannot find id for {obj}!")
            exit(-1)

        data_cam[name] = {}
        data_cam[name]["0"] = {
            "t": position.tolist(),
            "rot": rotation.tolist(),
            "box": bbox,
            "class": class_id,
        }

        obj_position = np.dot(-rotation.T, position)
        obj_rotation = rotation.T
        obj_position = center_object_origin(obj, obj_position)

        data_obj[name] = {}
        data_obj[name]["0"] = {
            "t": obj_position.tolist(),
            "rot": obj_rotation.tolist(),
            "box": bbox,
            "class": class_id
        }

        src = f"{base_path}/{object}/imgs/{name}.png"
        dst = f"{base_path}/{object}/rgb/{name}.png"
        image = cv2.imread(src)
        resized_image = cv2.resize(image, (width, height))
        cv2.imwrite(dst, resized_image)
        # shutil.copy2(src, dst)

    with open(f"{base_path}/{object}/gt_cam.json", 'w', encoding='utf-8') as f:
        json.dump(data_cam, f, ensure_ascii=False, indent=2)

    with open(f"{base_path}/{object}/gt_obj.json", 'w', encoding='utf-8') as f:
        json.dump(data_obj, f, ensure_ascii=False, indent=2)

    cnt = cnt + 1
    print(f"[INFO] Created gt_cam.json and gt_obj.json for '{object}'!")
    print(f"{GREEN}[SUCC]{RESET} Processed {cnt}/{len(recordings)}!")
    print("--------------------------------------------------")

print(f"Successfully extracted gt information for {cnt} records!")

# objects = ["book", "book1", "cabinet", "cabinet1", "can", "can1", "chair", "chair1","chair2", "doll", "doll1", "table"]
# # objects = ["book", "cabinet1", "can1", "chair1", "doll1"]
# for object in objects:
#     with open(f"files/{object}/{object}.txt") as file:
#         lines = [line.rstrip() for line in file]

#     def getPosition(content):
#         return [float(content[1]), float(content[2]), float(content[3])]

#     def getRotation(content):
#         return [float(content[4]), float(content[5]), float(content[6]), float(content[7])]

#     data_cam = {}
#     data_obj = {}
#     for line in lines:
#         content = line.split(" ")
#         name = content[0].split("/")[1].split(".")[0]
#         position = np.array(getPosition(content))
#         rotation = Rotation.from_quat(getRotation(content)).as_matrix()

#         # if not os.path.isdir(f"files/{object}/test"):
#         #     print(f"Test folder not present, skipping {object}")
#         # if not is_image_present(f"files/{object}/test", content[0].split("/")[1]):
#         #     continue

#         data_cam[name] = {}
#         data_cam[name]["0"] = {
#             "t": position.tolist(),
#             "rot": rotation.tolist(),
#             "box": [],
#             "class": 1
#         }

#         position_transf = np.dot(-rotation.T, position)

#         data_obj[name] = {}
#         data_obj[name]["0"] = {
#             "t": position_transf.tolist(),
#             "rot": rotation.T.tolist(),
#             "box": [],
#             "class": 1
#         }

#     with open(f"files/{object}/gt_cam.json", 'w', encoding='utf-8') as f:
#         json.dump(data_cam, f, ensure_ascii=False, indent=2)

#     with open(f"files/{object}/gt_obj.json", 'w', encoding='utf-8') as f:
#         json.dump(data_obj, f, ensure_ascii=False, indent=2)




