import json
import sys
import os

# Add the parent directory to sys.path, otherwise 'logger' from 'util' will be not found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from util import logger
import os
import numpy as np
from InferenceEngine import InferenceEngine
from helper import Args, controls, Pose, Rotation_, Translation_, IMG_WIDTH, IMG_HEIGHT
from PIL import Image
from scipy.spatial.transform import Rotation
import copy


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


STOP_THREADS = False


# display image in pygame
def display_img(frame: numpy.array, display: pygame.Surface):
    # make a pygame surface
    image = pygame.surfarray.make_surface(frame)
    # image has to be prepared for pygame screen with: rotation, flipping, scaling
    rotated_image = pygame.transform.rotate(image, 270)
    flipped_image = pygame.transform.flip(rotated_image, True, False)
    scaled_image = pygame.transform.scale(flipped_image, (IMG_WIDTH, IMG_HEIGHT))
    # display image in pygame window
    display.blit(scaled_image, (0, 0))


base_path: str = "demo/13_03"
def store_pose(pose: Pose, frame: int):
    name = "frame" + str(frame) + ".png"

    line = name + " " + str(pose.t.x) + " " + str(pose.t.y) + " " + str(
        pose.t.z) + " " + str(pose.R.x) + " " + str(pose.R.y) + " " + str(
        pose.R.z) + " " + str(pose.R.w) + "\n"
    
    gt_file.write(line)


def store_img(img: np.ndarray, frame: int):
    if img is None or img.size == 0: return

    # path = f"{base_path}/imgs/"
    path = rgb_folder
    name = "frame" + str(frame) + ".png"
    # img.save(os.path.join(name))
    cv2.imwrite(os.path.join(path, name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


last_pose_lock = threading.Lock()
last_pose: Pose = None
last_pose_time = None # Seconds
def pose_thread(pose: PoseStamped):
    global last_pose, last_pose_lock, last_pose_time

    # If lock already acquired, discard current pose
    if not last_pose_lock.acquire(blocking=False):
        return

    try:
        # if not pose or pose.header.seq < last_pose.header.seq: # Sanity check -> skip pose if older than last one
        #     last_pose_lock.release()
        #     return

        pose.header.frame_id = "world"
        t = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])
        R = Rotation.from_quat([pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z,
                                pose.pose.orientation.w]).as_matrix()

        last_pose = Pose(t, R, pose.header.seq, pose.header.stamp)
        last_pose_time = pose.header.stamp.secs + pose.header.stamp.nsecs * 1e-9
    finally:
        last_pose_lock.release()


last_frame_lock = threading.Lock()
last_frame: Image = None
last_frame_time = 0 # Seconds
def image_thread(container: av.container.InputContainer):
    global last_frame ,last_frame_lock, last_frame_time

    try:
        # skip first 300 frames in order to avoid delay
        # frame_skip = 300

        frame: av.video.frame.VideoFrame
        for frame in container.decode(video=0):
            if STOP_THREADS == True:
                return
    
            # if 0 < frame_skip:
            #     frame_skip = frame_skip - 1
            #     continue

            # If lock already acquired, discard current image
            if not last_frame_lock.acquire(blocking=False):
                continue

            # if frame.time < last_frame_time: # Sanity check -> skip frame if older than last one
            #     last_frame_lock.release()
            #     continue

            # tmp = np.array(frame.to_image())  # Convert frame to numpy array
            # last_frame = cv2.resize(tmp, (IMG_WIDTH, IMG_HEIGHT))  # Resize image appropriately for inference
            # last_frame = Image.fromarray(last_frame).convert("RGB")
            last_frame = frame.to_image(width=IMG_WIDTH, height=IMG_HEIGHT)
            ## TODO: Use frame.to_ndarray() instead of frame.to_image()!!
            # Get the timestamp of the frame
            last_frame_time = frame.time
            last_frame_lock.release()
    except e:
        logger.err(str(e))



frame_number: int = 0
pose_pub = None
def inference_thread(image_display: pygame.Surface):
    global last_frame
    global frame_number
    global engine
    global last_pose
    global IMG_WIDTH, IMG_HEIGHT
    global pose_pub

    while True:

        if STOP_THREADS == True:
            return

        # Immediately get last recorded ground-truth pose and image of drone
        gt: Pose
        with last_pose_lock:
            if last_pose is not None:
                gt = copy.deepcopy(last_pose)  # Ensure to get a deep-copy of the last pose, so that we don't hold a reference
                # if last_pose and last_pose.id >= gt.id: # Only accept poses that are younger than previous
                #     continue
            else:
                if STOP_THREADS == True:
                    return
                # logger.warn(f"[{frame_number:04d}] Got no pose, skipping inference on drone image ...")
                continue  # If no pose recorded, skip

        frame: Image
        with last_frame_lock:
            if last_frame is not None:
                frame = copy.deepcopy(last_frame)
            else:
                if STOP_THREADS == True:
                    return
                logger.warn(f"[{frame_number:04d}] Got no frame, skipping inference on drone image ...")
                continue

        if STOP_THREADS == True:
            return

        start_time = time.time()

        # Store recorded image and ground-truth data
        store_img(np.array(frame), frame_number)
        store_pose(gt, frame_number)

        # Do inference ...
        res: dict = {}
        try:
            res, inf_time, poet_time, t_rmse, R_rmse = engine.inference(frame, gt, frame_number)
        except Exception as error:
            logger.warn(error.with_traceback())
            frame_number += 1
            continue

        # Display annotated image
        if res and res[0] and "img" in res[0]:
            display_img(res[0]["img"], image_display)

            # path = bbox_folder
            # name = f"frame{frame_number}_1.png"
            # cv2.imwrite(os.path.join(path, name), cv2.cvtColor(res[0]["img"], cv2.COLOR_RGB2BGR))
        else:
            display_img(np.array(frame), image_display)
            logger.warn(f"[{frame_number:04d}] Got no prediction for frame ...")
            frame_number += 1
            continue

        # Publish prediction as ros message
        t = Translation_()
        if res:
            for key in res:
                t = Translation_(res[key]["t"][0], res[key]["t"][1], res[key]["t"][2])
                R = Rotation_(res[key]["rot"])
                pose_stamped = PoseStamped()
                pose_stamped.header.frame_id = "world"
                pose_stamped.pose.position.x = t.x
                pose_stamped.pose.position.y = t.y
                pose_stamped.pose.position.z = t.z

                pose_stamped.pose.orientation.x = R.x
                pose_stamped.pose.orientation.y = R.y
                pose_stamped.pose.orientation.z = R.z
                pose_stamped.pose.orientation.w = R.w

                pose_pub.publish(pose_stamped)

        # print(f"total: {time.time() - start_time}s")
        #txt = f"[{frame_number:04d}] Total: {time.time() - start_time:.4f}s | PoEt: {poet_time:>.4f}s | RMSE (t): {t_rmse:.4f}, (R): {R_rmse:.4f}"
        txt = f"[{frame_number:04d}] Total: {time.time() - start_time:.4f}s | PoEt: {poet_time:>.4f}s | x: {gt.t.x:.4f} | y: {gt.t.y:.4f} | z : {gt.t.z:.4f} | RMSE (t): {t_rmse:.4f}, (R): {R_rmse:.4f}"
        logger.succ(txt)
        log_file.write(txt + "\n")

        frame_number += 1

        if STOP_THREADS == True:
            return 


log_file = None
pred_file = None
gt_file = None
rgb_folder = None
bbox_folder = None


def createFolderStructure():
    global log_file, pred_file, gt_file, rgb_folder, bbox_folder

    i = 0
    while os.path.exists(f"{base_path}/imgs_%s" % i):
        i += 1

    rgb_folder = f"{base_path}/imgs_%s/rgb" % i
    os.makedirs(rgb_folder)

    bbox_folder = f"{base_path}/imgs_%s/bbox" % i
    os.makedirs(bbox_folder)

    pred_file = open(f"{base_path}/imgs_%s/pred.json" % i, "w")
    gt_file = open(f"{base_path}/imgs_%s/gt_cam.txt" % i, "w")
    log_file = open(f"{base_path}/imgs_%s/log.txt" % i, "w")

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
    args.resume = "/home/wngicg/repos/poet/results/finetune/drone_data/2024-11-27_09_19_53/checkpoint.pth"
    #args.resume = "/home/wngicg/Desktop/repos/poet/results/train/2024-10-06_12_31_12/checkpoint.pth"
    #args.resume = "/media/wngicg/USB-DATA/repos/poet/results_doll/train/2024-10-13_14_09_21/checkpoint.pth"
    args.device = "cuda"

    #args.dino_caption = "black cabinet."
    args.dino_caption = "human doll."
    args.dino_args = "models/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    args.dino_checkpoint = "models/groundingdino/weights/groundingdino_swint_ogc.pth"
    args.dino_box_threshold = 0.35
    args.dino_txt_threshold = 0.25
    args.dino_cos_cim = 0.9
    args.dino_bbox_viz = False


    #     parser.add_argument('--dino_caption', default=None, type=str, help='Caption for Grounding DINO object detection')
    # parser.add_argument('--dino_args', default="models/groundingdino/config/GroundingDINO_SwinT_OGC.py", type=str, help='Args for Grounding DINO backbone')
    # parser.add_argument('--dino_checkpoint', default="models/groundingdino/weights/groundingdino_swint_ogc.pth", type=str, help='Checkpoint for Grounding DINO backbone')
    # parser.add_argument('--dino_box_threshold', default=0.35, type=float, help='Bounding Box threshold for Grounding DINO')
    # parser.add_argument('--dino_txt_threshold', default=0.25, type=float, help='Text threshold for Grounding DINO')
    # parser.add_argument('--dino_cos_sim', default=0.9, type=float, help='Cosine similarity for matching Grounding DINO predictions to labels')
    # parser.add_argument('--dino_bbox_viz', default=False, type=bool, help='Visualize Grounding DINO bounding box predictions and labels')

    # ---------------------------------------------------------------------------------------------

    engine = InferenceEngine(args, draw=True)
    logger.succ("Initialized PoET inference engine!")

    pygame.init()
    pygame.display.init()

    rospy.init_node('poet_demo_node')

    image_display = pygame.display.set_mode((IMG_WIDTH, IMG_HEIGHT))
    pygame.display.set_caption('Camera stream (live)')

    # display icg icon instead of pygame icon
    # icon = pygame.image.load(ICG_LOGO_PATH)
    # pygame.display.set_icon(icon)

    drone = tellopy.Tello()
    drone.set_loglevel(1)  # log level WARN
    speed = 30

    img_thread = None
    pose_sub = None
    try:
        drone.connect()
        drone.wait_for_connection(60.0)

        retry = 3
        container: av.container.InputContainer = None

        # get video stream from drone
        while container is None and 0 < retry:
            retry -= 1
            try:
                container = av.open(drone.get_video_stream())
            except av.AVError as ave:
                print(ave)
                print('Retrying getting tello video stream ...')

        createFolderStructure()

        img_thread = threading.Thread(target=image_thread, args=(container,))
        img_thread.start()
        logger.succ("Image thread started!")

        while last_frame is None:
            time.sleep(0.5)

        inf_thread = threading.Thread(target=inference_thread, args=(image_display,))
        inf_thread.start()
        logger.succ("Inference thread started!")

        pose_topic = "/mocap_node/tello/pose"
        pose_sub = rospy.Subscriber(pose_topic, PoseStamped, callback=pose_thread, queue_size=100)
        logger.succ("Pose thread started!")

        pose_pub = rospy.Publisher("/pose_out", PoseStamped)

        while True:
            time.sleep(0.01)  # loop with pygame.event.get() is too tight w/o some sleep

            # update image displayed in pygame window
            pygame.display.update()
            # movement processing/keyboard control
            for e in pygame.event.get():
                # check if any key was pressed
                if e.type == pygame.locals.KEYDOWN:
                    # print('+' + pygame.key.name(e.key))
                    keyname = pygame.key.name(e.key)
                    # check if pressed key was escape (shutdown program)
                    if keyname == 'escape':
                        drone.quit()
                        raise Exception("Termination signal received ...")
                    
                    # check if pressed key was one of controls dict from above
                    if keyname in controls:
                        key_handler = controls[keyname]
                        if type(key_handler) == str:
                            getattr(drone, key_handler)(speed)
                        else:
                            key_handler(drone, speed)

                # check if pressed key was released (set speed to zero)
                elif e.type == pygame.locals.KEYUP:
                    # print('-' + pygame.key.name(e.key))
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
        logger.warn("Exiting PoET demo script!")

        STOP_THREADS = True

        img_thread.join()
        inf_thread.join()

        logger.info(f"Predicted {len(engine.results)} poses")
        logger.info("Saving predicted poses as json ...")

        # Remove images from results! We don't want to save them!
        for frame in engine.results:
            for i in engine.results[frame]:
                if 'img' in engine.results[frame][i]:
                    engine.results[frame][i]['t'] = engine.results[frame][i]['t'].tolist()
                    engine.results[frame][i]['rot'] = engine.results[frame][i]['rot'].tolist()
                    engine.results[frame][i]['box'] = engine.results[frame][i]['box'].tolist()
                    del engine.results[frame][i]['img']

        json.dump(engine.results, pred_file, indent=4)

        logger.warn('Shutting down connection to drone...')
        drone.quit()
        exit(1)
