import sys
import os

# Add the parent directory to sys.path, otherwise 'logger' from 'util' will be not found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from util import logger
from models import build_model
import time
import os
import supervision as sv
import numpy as np
from helper import Args, controls
from PIL import Image

import av
import cv2
import numpy
import time
import tellopy
import pygame
import pygame.locals
import threading
import rospy
from geometry_msgs.msg import PoseStamped

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def mean_translation(data):
    translations = np.array([entry['t'] for entry in data])  # Extract all translations
    mean_t = np.mean(translations, axis=0)  # Compute mean
    return mean_t


def mean_rotation(data):
    rotations = np.array([entry['rot'] for entry in data])  # Extract rotations
    mean_rot = np.mean(rotations, axis=0)  # Average element-wise

    # SVD to ensure valid rotation matrix
    U, _, Vt = np.linalg.svd(mean_rot)
    mean_rot_valid = np.dot(U, Vt)  # Ensure orthogonality

    # Correct sign if determinant is negative
    if np.linalg.det(mean_rot_valid) < 0:
        U[:, -1] *= -1
        mean_rot_valid = np.dot(U, Vt)

    return mean_rot_valid


class GroundTruth:
    def __init__(self, t, R):
        self.t = t
        self.R = R


class InferenceEngine:
    def __init__(self, args, draw: bool = False):
        self.args = args
        self.device = torch.device(args.device)
        self.model, self.criterion, self.matcher = build_model(args)
        self.model.to(self.device)
        self.model.eval()

        # Load model weights
        self.checkpoint = torch.load(args.resume, map_location='cpu')
        self.model.load_state_dict(self.checkpoint['model'], strict=False)

        self.img_id = 0
        self.results = {}

        self.draw = draw

    def rmse_rotation(self, rot_est, rot_gt):
        rot = np.matmul(rot_est, rot_gt.T)
        trace = np.trace(rot)
        if trace < -1.0:
            trace = -1
        elif trace > 3.0:
            trace = 3.0
        angle_diff = np.degrees(np.arccos(0.5 * (trace - 1)))
        return angle_diff

    @staticmethod
    def rmse_translation(array1, array2):
        """Calculate the root mean square error (RMSE) between two arrays."""
        return np.sqrt(np.sum(np.square((array1 - array2))))

    @staticmethod
    def transform_to_cam(R, t):
        return R.T, np.dot(-R.T, t)

    @staticmethod
    def transform_bbox(bbox, img_size):
        """
        Convert normalized cxcywh bounding box coordinates to pixel coordinates in xyxy format.

        Args:
        - normalized_bbox: Tuple of (cx, cy, w, h) normalized to [0, 1]
        - image_size: Tuple of (image_width, image_height) in pixels

        Returns:
        - Tuple of (x1_pixel, y1_pixel, x2_pixel, y2_pixel)
        """
        cx_normalized, cy_normalized, w_normalized, h_normalized = bbox
        image_width, image_height = img_size

        # Convert normalized values to pixel values
        x_center_pixel = cx_normalized * image_width
        y_center_pixel = cy_normalized * image_height
        width_pixel = w_normalized * image_width
        height_pixel = h_normalized * image_height

        # Calculate top-left and bottom-right corners
        x1_pixel = x_center_pixel - (width_pixel / 2)
        y1_pixel = y_center_pixel - (height_pixel / 2)
        x2_pixel = x_center_pixel + (width_pixel / 2)
        y2_pixel = y_center_pixel + (height_pixel / 2)

        return [x1_pixel, y1_pixel, x2_pixel, y2_pixel]

    def inference(self, frame: np.ndarray, gt: GroundTruth, verbose: bool = True):
        r"""
        Args:
          frame (np.ndarray): List of images to do inference on.
          gt: (GroundTruth): Ground truth pose from tracking system.
          verbose (bool): Print runtime, results, RMSEs, etc.
        """

        ## TODO: Check if w, h is correct
        h, w, _ = frame.shape
        if w != 640 or h != 480:
            logger.err("Given image has not required size (640, 480)!")
            return None

        frame_orig = frame
        # PoET expects input in (bs, c, h, w) format!
        frame = np.moveaxis(frame, 2, 0) #(c, h, w)

        frame = np.expand_dims(frame, axis=0)  # Add dimension in the front for list of tensors -> (1, c, h, w)
        frame = torch.from_numpy(frame).to(self.device).float()

        start_ = time.time()
        outputs, n_boxes_per_sample = self.model(frame, None)
        poet_time = time.time() - start_

        val = False
        for i in outputs["pred_boxes"].detach().cpu().tolist():
            for a in i:
                for b in a:
                    if b != -1:
                        val = True

        if not val:
            logger.warn("No prediction ...")
            return None

        if outputs is None:
            return None

        # Iterate over all the detected predictions
        result = []
        for d in range(n_boxes_per_sample[0]):
            pred_t = np.array(outputs['pred_translation'][0][d].detach().cpu().tolist())
            pred_rot = np.array(outputs['pred_rotation'][0][d].detach().cpu().tolist())
            pred_box = np.array(outputs['pred_boxes'][0][d].detach().cpu().tolist())
            pred_class = np.array(outputs['pred_classes'][0][d].detach().cpu().tolist())

            R, t = self.transform_to_cam(pred_rot, pred_t)

            result.append({
                "t": t,
                "rot": R,
                "box": pred_box,  # format: cxcywh
                "class": pred_class
            })

            if self.draw:
                # Draw predicted bounding box
                detections = sv.Detections(
                    xyxy=np.array([self.transform_bbox(pred_box, (640, 480))]))  # transform normalized cxcywh to xyxy
                box_annotator = sv.BoxAnnotator()
                img = Image.fromarray(frame_orig)
                annotated_frame = box_annotator.annotate(scene=img, detections=detections)
                result[d]["img"] = np.array(annotated_frame)

        if not result: return None

        t = mean_translation(result)
        R = mean_rotation(result)

        t_rmse = self.rmse_translation(t, gt.t)
        R_rmse = self.rmse_rotation(R, gt.R)

        if verbose:
            logger.succ(
                f"[{self.img_id:04d}] Total: {time.time() - start_:.4f}s | PoEt: {poet_time:.4f}s | x: {t[0]} | y: {t[1]} | z : {t[2]} | RMSE (t): {t_rmse:.4f}, (R): {R_rmse:.4f}")

        self.results[self.img_id] = result
        self.img_id += 1
        return result


# display image in pygame
def display_img(frame: numpy.array, display: pygame.Surface):
    # make a pygame surface
    image = pygame.surfarray.make_surface(frame)
    # image has to be prepared for pygame screen with: rotation, flipping, scaling
    rotated_image = pygame.transform.rotate(image, 270)
    flipped_image = pygame.transform.flip(rotated_image, True, False)
    scaled_image = pygame.transform.scale(flipped_image, (IMG_ORIG_WIDTH, IMG_ORIG_HEIGHT))
    # display image in pygame window
    display.blit(scaled_image, (0, 0))


counter: int = 0
frame_number: int = 0
base_path: str = "fly/"

frame_glob: int = 0


def store_pose(pose: PoseStamped):
    global frame_number

    name = "frame" + str(frame_number) + ".png"

    line = name + " " + str(pose.pose.position.x) + " " + str(pose.pose.position.y) + " " + str(
        pose.pose.position.z) + " " + str(pose.pose.orientation.x) + " " + str(pose.pose.orientation.y) + " " + str(
        pose.pose.orientation.z) + " " + str(pose.pose.orientation.w) + "\n"

    out_file = open(os.path.join(base_path, f"gt_cam.txt"), "a")
    out_file.write(line)


def store_img(img: numpy.array, number: int):
    if img is None: return

    path = f"{base_path}/imgs/"
    name = "frame" + str(number) + ".png"
    cv2.imwrite(os.path.join(path, name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def pose_thread(pose: PoseStamped):
    global counter
    global frame_number
    counter += 1

    if counter % 10 == 0 and frame_glob is not None:
        print("received pose: ", counter)

        store_img(frame_glob, frame_number)
        store_pose(pose)

        frame_number += 1
        pose.header.frame_id = "world"


def image_thread(image_display: pygame.Surface, container):
    global frame_glob
    global frame_number
    global engine

    counter = 0

    # skip first 300 frames in order to avoid delay
    frame_skip = 300
    for frame in container.decode(video=0):
        if 0 < frame_skip:
            frame_skip = frame_skip - 1
            continue

        # if counter % 10:
        #   counter += 1
        #   continue

        counter += 1

        start_time = time.time()

        tmp = numpy.array(frame.to_image())
        frame_glob = cv2.resize(tmp, (640, 480))
        # store_img(frame_glob, frame_number)
        frame_number += 1

        gt = GroundTruth(np.array([0, 0, 0]), np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]))
        res = None
        try:
            res = engine.inference(frame_glob, gt)
        except Exception as error:
            logger.warn(error.with_traceback())

        # display image
        if res and res[0] and "img" in res[0]:
            display_img(res[0]["img"], image_display)
        else:
            display_img(frame_glob, image_display)

        # we want to skip the frames since the last processed frame in order to avoid delay
        if frame.time_base < 1.0 / 60:
            time_base = 1.0 / 60
        else:
            time_base = frame.time_base
        frame_skip = int((time.time() - start_time) / time_base)


IMG_ORIG_WIDTH = 640
IMG_ORIG_HEIGHT = 480

engine: InferenceEngine = None

if __name__ == "__main__":
    args = Args()

    args.enc_layers = 5
    args.dec_layers = 5
    args.nheads = 16
    args.batch_size = 16
    args.eval_batch_size = 16
    args.n_classes = 16
    args.class_mode = "agnostic"

    args.backbone = "dinoyolo"
    args.lr_backbone = 0.0
    args.backbone_cfg = "./configs/ycbv_yolov4-csp.cfg"
    args.backbone_weights = "/home/wngicg/Desktop/repos/ycbv_yolo_weights.pt"
    args.dataset_path = "/home/wngicg/Desktop/repos/datasets/custom"
    args.class_info = "/annotations/custom_classes.json"
    args.model_symmetry = "/annotations/custom_symmetries.json"

    # args.train_set = "train"
    # args.eval_set = "val"
    # args.test_set = "test"

    args.eval_interval = 1
    # args.output_dir = "train/"
    args.bbox_mode = "backbone"
    args.dataset = "custom"
    args.grayscale = True  # Assuming this is a flag, set to True
    args.rgb_augmentation = True  # Assuming this is a flag, set to True
    # args.translation_loss_coef = 2.0
    # args.rotation_loss_coef = 1.0
    # args.epochs = 0
    # args.lr = 0.000035
    # args.lr_drop = 50
    # args.gamma = 0.1
    args.resume = "/home/wngicg/Desktop/repos/poet/results/train/2024-10-06_12_31_12/checkpoint.pth"
    args.device = "cuda"

    # ---------------------------------------------------------------------------------------------

    engine = InferenceEngine(args, draw=True)
    logger.succ("Initialized PoET inference engine!")

    pygame.init()
    pygame.display.init()

    rospy.init_node('poet_demo_node')

    image_display = pygame.display.set_mode((IMG_ORIG_WIDTH, IMG_ORIG_HEIGHT))
    pygame.display.set_caption('Camera stream (live)')

    # display icg icon instead of pygame icon
    # icon = pygame.image.load(ICG_LOGO_PATH)
    # pygame.display.set_icon(icon)

    drone = tellopy.Tello()
    drone.set_loglevel(1)  # log level WARN
    speed = 30

    try:
        drone.connect()
        drone.wait_for_connection(60.0)

        retry = 3
        container = None

        # get video stream from drone
        while container is None and 0 < retry:
            retry -= 1
            try:
                container = av.open(drone.get_video_stream())
            except av.AVError as ave:
                print(ave)
                print('Retrying getting tello video stream ...')

        key_thread = threading.Thread(target=image_thread, args=(image_display, container))
        key_thread.start()
        logger.succ("Image thread started!")

        pose_topic = "/mocap_node/tello/pose"
        pose_sub = rospy.Subscriber(pose_topic, PoseStamped, callback=pose_thread)

        while True:
            time.sleep(0.01)  # loop with pygame.event.get() is too tight w/o some sleep

            # update image displayed in pygame window
            pygame.display.update()
            # movement processing/keyboard control
            for e in pygame.event.get():
                # check if any key was pressed
                if e.type == pygame.locals.KEYDOWN:
                    print('+' + pygame.key.name(e.key))
                    keyname = pygame.key.name(e.key)
                    # check if pressed key was escape (shutdown program)
                    if keyname == 'escape':
                        drone.quit()
                        exit(0)
                    # check if pressed key was one of controls dict from above
                    if keyname in controls:
                        key_handler = controls[keyname]
                        if type(key_handler) == str:
                            getattr(drone, key_handler)(speed)
                        else:
                            key_handler(drone, speed)

                # check if pressed key was released (set speed to zero)
                elif e.type == pygame.locals.KEYUP:
                    print('-' + pygame.key.name(e.key))
                    keyname = pygame.key.name(e.key)
                    if keyname in controls:
                        key_handler = controls[keyname]
                        if type(key_handler) == str:
                            getattr(drone, key_handler)(0)
                        else:
                            key_handler(drone, 0)

    except e:
        logger.err(str(e))
    finally:
        logger.warn('Shutting down connection to drone...')
        drone.quit()
        exit(1)
