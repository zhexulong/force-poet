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
import torch

warnings.filterwarnings("ignore", category=DeprecationWarning)


STOP_THREADS = False


def draw_contact_forces_on_image(rgb_image, force_matrix, poses_6d, camera_intrinsics, scale_factor=0.05, min_force_magnitude=0.1):
    """
    在RGB图像上绘制接触力向量
    
    Args:
        rgb_image: RGB图像 (numpy array)
        force_matrix: 力矩阵 [N, N, 3] - N个物体之间的力
        poses_6d: 物体6D位姿 [N, 7] - position + quaternion
        camera_intrinsics: 相机内参矩阵
        scale_factor: 力向量可视化缩放因子
        min_force_magnitude: 最小显示力大小
    """
    if force_matrix is None or poses_6d is None:
        return rgb_image
    
    img = rgb_image.copy()
    h, w = img.shape[:2]
    
    # 获取有效物体数量
    num_objects = force_matrix.shape[0]
    
    # 颜色列表（BGR格式）
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for i in range(num_objects):
        for j in range(num_objects):
            if i == j:  # 跳过自身作用力
                continue
                
            force_vector = force_matrix[i, j]  # 物体j对物体i的力
            force_magnitude = np.linalg.norm(force_vector)
            
            if force_magnitude < min_force_magnitude:
                continue
            
            # 获取物体位置
            pos_i = poses_6d[i][:3]  # 物体i的位置
            
            # 力向量终点位置
            end_pos = pos_i + force_vector * scale_factor
            
            # 投影到2D图像
            try:
                points_3d = np.array([pos_i, end_pos]).reshape(-1, 3)
                rvec = np.zeros(3)
                tvec = np.zeros(3)
                
                points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_intrinsics, None)
                points_2d = points_2d.reshape(-1, 2).astype(int)
                
                start_point = tuple(points_2d[0])
                end_point = tuple(points_2d[1])
                
                # 检查点是否在图像范围内
                if (0 <= start_point[0] < w and 0 <= start_point[1] < h and
                    0 <= end_point[0] < w and 0 <= end_point[1] < h):
                    
                    color = colors[i % len(colors)]
                    
                    # 绘制力向量箭头
                    cv2.arrowedLine(img, start_point, end_point, color, 2, tipLength=0.3)
                    
                    # 添加力大小标签
                    force_text = f"{force_magnitude:.2f}N"
                    cv2.putText(img, force_text, end_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
            except Exception as e:
                logger.warning(f"Failed to draw force vector {i}->{j}: {e}")
                continue
    
    return img


def draw_6d_poses_on_image(rgb_image, poses_6d, camera_intrinsics, axis_length=0.05, obj_ids=None):
    """
    在RGB图像上绘制6D位姿坐标轴
    
    Args:
        rgb_image: RGB图像 (numpy array)
        poses_6d: 物体6D位姿 [N, 7] - position + quaternion
        camera_intrinsics: 相机内参矩阵
        axis_length: 坐标轴长度
        obj_ids: 物体ID列表
    """
    if poses_6d is None or len(poses_6d) == 0:
        return rgb_image
    
    img = rgb_image.copy()
    h, w = img.shape[:2]
    
    # 坐标轴颜色 (BGR格式): X-红, Y-绿, Z-蓝
    axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    axis_labels = ['X', 'Y', 'Z']
    
    for obj_idx, pose in enumerate(poses_6d):
        position = pose[:3]
        quaternion = pose[3:]  # [qx, qy, qz, qw]
        
        # 转换四元数为旋转矩阵
        r = Rotation.from_quat(quaternion)
        rotation_matrix = r.as_matrix()
        
        # 定义坐标轴向量
        axes = np.array([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]])
        
        # 旋转坐标轴
        rotated_axes = rotation_matrix @ axes.T
        
        # 计算坐标轴终点
        axis_points = position.reshape(1, 3) + rotated_axes.T
        all_points = np.vstack([position.reshape(1, 3), axis_points])
        
        try:
            # 投影到2D
            rvec = np.zeros(3)
            tvec = np.zeros(3)
            points_2d, _ = cv2.projectPoints(all_points, rvec, tvec, camera_intrinsics, None)
            points_2d = points_2d.reshape(-1, 2).astype(int)
            
            origin_2d = tuple(points_2d[0])
            axis_ends_2d = points_2d[1:4]
            
            # 检查原点是否在图像范围内
            if 0 <= origin_2d[0] < w and 0 <= origin_2d[1] < h:
                # 绘制原点
                cv2.circle(img, origin_2d, 3, (255, 255, 255), -1)
                
                # 添加物体ID标签
                if obj_ids is not None:
                    label = str(obj_ids[obj_idx])
                    cv2.putText(img, label, (origin_2d[0] + 5, origin_2d[1] - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 绘制坐标轴
                for axis_idx in range(3):
                    end_point = tuple(axis_ends_2d[axis_idx])
                    if 0 <= end_point[0] < w and 0 <= end_point[1] < h:
                        cv2.line(img, origin_2d, end_point, axis_colors[axis_idx], 2)
                        # 添加轴标签
                        cv2.putText(img, axis_labels[axis_idx], 
                                   (end_point[0] + 2, end_point[1] + 2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, axis_colors[axis_idx], 1)
                        
        except Exception as e:
            logger.warning(f"Failed to draw pose for object {obj_idx}: {e}")
            continue
    
    return img


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


base_path: str = "demo/26_03"
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
    # if not last_pose_lock.acquire(blocking=False):
    #     return

    # try:
        # if not pose or pose.header.seq < last_pose.header.seq: # Sanity check -> skip pose if older than last one
        #     last_pose_lock.release()
        #     return

    pose.header.frame_id = "world"
    t = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])
    R = Rotation.from_quat([pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z,
                            pose.pose.orientation.w]).as_matrix()

    last_pose = Pose(t, R, pose.header.seq, pose.header.stamp)
    last_pose_time = pose.header.stamp.secs + pose.header.stamp.nsecs * 1e-9
    # finally:
        # last_pose_lock.release()


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
            # if not last_frame_lock.acquire(blocking=False):
            #     continue

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
            # last_frame_lock.release()
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
        # with last_pose_lock:
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
        # with last_frame_lock:
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
        force_matrix = None
        poses_6d = None
        try:
            res, inf_time, poet_time, t_rmse, R_rmse = engine.inference(frame, gt, frame_number)
            
            # Extract force prediction if available
            if hasattr(engine, 'last_outputs') and engine.last_outputs is not None:
                outputs = engine.last_outputs
                if 'pred_force_matrix' in outputs:
                    pred_force_matrix = outputs['pred_force_matrix'][0].detach().cpu().numpy()  # [n_queries, n_queries, 3]
                    force_matrix = pred_force_matrix
                    logger.info(f"[{frame_number:04d}] Force matrix shape: {force_matrix.shape}")
                
                # Extract poses for visualization
                if res and len(res) > 0:
                    poses_6d = []
                    obj_ids = []
                    for obj_idx in res:
                        obj_data = res[obj_idx]
                        t = obj_data['t']  # translation
                        R = obj_data['rot']  # rotation matrix
                        
                        # Convert rotation matrix to quaternion
                        r = Rotation.from_matrix(R)
                        quat = r.as_quat()  # [qx, qy, qz, qw]
                        
                        # Combine translation and quaternion
                        pose_6d = np.concatenate([t, quat])
                        poses_6d.append(pose_6d)
                        obj_ids.append(obj_data.get('class', obj_idx))
                    
                    poses_6d = np.array(poses_6d) if poses_6d else None
                    
        except Exception as error:
            logger.warn(f"Inference error: {error}")
            frame_number += 1
            continue

        # Enhanced visualization with force vectors and poses
        display_image = None
        if res and res[0] and "img" in res[0]:
            # Get the annotated image from inference
            display_image = res[0]["img"]
            
            # Add 6D pose visualization
            if poses_6d is not None:
                # Create a simple camera intrinsics matrix for visualization
                # These values should match your actual camera parameters
                cam_K = np.array([[525.0, 0, 320.0],
                                 [0, 525.0, 240.0],
                                 [0, 0, 1.0]])
                
                display_image = draw_6d_poses_on_image(
                    display_image, poses_6d, cam_K, axis_length=0.05, obj_ids=obj_ids)
            
            # Add force visualization
            if force_matrix is not None and poses_6d is not None:
                display_image = draw_contact_forces_on_image(
                    display_image, force_matrix, poses_6d, cam_K, 
                    scale_factor=0.05, min_force_magnitude=0.1)
                    
            display_img(display_image, image_display)
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

    # Basic model configuration
    args.enc_layers = 5
    args.dec_layers = 5
    args.nheads = 16
    args.batch_size = 8
    args.eval_batch_size = 8
    args.n_classes = 21
    args.class_mode = "specific"

    # Backbone configuration for Isaac Sim dataset
    args.backbone = "maskrcnn"
    args.lr_backbone = 1e-5
    args.backbone_cfg = "./configs/ycbv_rcnn.yaml"
    args.backbone_weights = "./weights/ycbv_maskrcnn_checkpoint.pth.tar"
    
    # Dataset configuration
    args.dataset_path = "../isaac_sim_poet_dataset_force"
    args.class_info = "/annotations/classes.json"
    args.model_symmetry = "/annotations/isaac_sim_symmetries.json"

    # Force prediction configuration
    args.use_force_prediction = True
    args.force_loss_coef = 1.0
    
    # Evaluation settings
    args.eval_interval = 1
    args.bbox_mode = "gt"  # Use ground truth bounding boxes for inference
    args.dataset = "custom"
    args.grayscale = False
    args.rgb_augmentation = True
    
    # Model checkpoint from Isaac Sim training results
    args.resume = "/data/gplong/force_map_project/w-poet/poet/results/isaac_sim_training/2025-07-26_05:51:59/checkpoint.pth"
    args.device = "cuda"

    # Demo-specific settings for object detection
    args.dino_caption = "objects"
    args.dino_args = "models/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    args.dino_checkpoint = "models/groundingdino/weights/groundingdino_swint_ogc.pth"
    args.dino_box_threshold = 0.35
    args.dino_txt_threshold = 0.25
    args.dino_cos_cim = 0.9
    args.dino_bbox_viz = False

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
