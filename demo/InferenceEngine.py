import numpy as np
import torch
import time
from helper import Rotation_, Translation_, Pose, Args, dimensions
from models import build_model
import sys, os
from PIL import Image
import torchvision.transforms.functional as F
import supervision as sv

# Add the parent directory to sys.path, otherwise 'logger' from 'util' will be not found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from util import logger

def mean_translation(data):
    translations = np.array([data[key]['t'] for key in data])  # Extract all translations
    mean_t = np.mean(translations, axis=0)  # Compute mean
    return mean_t


def mean_rotation(data):
    rotations = np.array([data[key]['rot'] for key in data])  # Extract rotations
    mean_rot = np.mean(rotations, axis=0)  # Average element-wise

    # SVD to ensure valid rotation matrix
    U, _, Vt = np.linalg.svd(mean_rot)
    mean_rot_valid = np.dot(U, Vt)  # Ensure orthogonality

    # Correct sign if determinant is negative
    if np.linalg.det(mean_rot_valid) < 0:
        U[:, -1] *= -1
        mean_rot_valid = np.dot(U, Vt)

    return mean_rot_valid

def get_center_of_object(object: str):
    """
    Returns the center of object (in pixels) of the given object.
    """
    h = dimensions[object]["h"]
    return np.array([0, 0, h / 2])

def bottom_object_origin(object: str, t: np.ndarray):
    """
    Centers the given origin (t) of an object.
    """
    center = get_center_of_object(object)
    return t - center

class InferenceEngine:
    def __init__(self, args: Args, draw: bool = False):
        self.args = args
        self.device = torch.device(args.device)
        self.model, self.criterion, self.matcher = build_model(args)
        self.model.to(self.device)
        self.model.eval()

        # Load model weights
        self.checkpoint = torch.load(args.resume, map_location='cpu')
        self.model.load_state_dict(self.checkpoint['model'], strict=False)

        self.results: dict = {}

        self.draw = draw

    @staticmethod
    def rmse_rotation(rot_est: np.ndarray, rot_gt: np.ndarray):
        rot = np.matmul(rot_est, rot_gt.T)
        trace = np.trace(rot)
        if trace < -1.0:
            trace = -1
        elif trace > 3.0:
            trace = 3.0
        angle_diff = np.degrees(np.arccos(0.5 * (trace - 1)))
        return angle_diff

    @staticmethod
    def rmse_translation(array1: np.ndarray, array2: np.ndarray):
        """Calculate the root mean squared error (RMSE) between two arrays."""
        return np.sqrt(np.sum(np.square((array1 - array2))))

    @staticmethod
    def transform_to_cam(R: np.ndarray, t: np.ndarray, obj: str):
        # TODO: Refactor Object
        t = bottom_object_origin(object=obj, t=t)
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

    def inference(self, frame_orig: Image, gt: Pose, frame_number: int, verbose: bool = True):
        r"""
        Args:
          frame_orig (np.ndarray): List of images to do inference on.
          gt: (Pose): Ground truth pose from tracking system.
          verbose (bool): Print runtime, results, RMSEs, etc.
        """
        global IMG_WIDTH, IMG_HEIGHT

        # h, w, _ = frame_orig.shape # nd.array
        w, h = frame_orig.size  # Image

        if w != IMG_WIDTH or h != IMG_HEIGHT:
            logger.err(f"Given image has not required size ({IMG_WIDTH} x {IMG_HEIGHT})!")
            return None

        frame = np.array(frame_orig)  # Image -> nd.array
        frame = F.to_tensor(frame).to("cuda")  # nd.array -> tensor.cuda.floatTensor
        frame = frame.unsqueeze(0)  #

        # PoET expects input in (bs, c, h, w) format!
        # frame = np.moveaxis(frame, 2, 0) #(c, h, w)

        # frame = np.expand_dims(frame, axis=0)  # Add dimension in the front for list of tensors -> (1, c, h, w)
        # frame = torch.from_numpy(frame).to(self.device).float()

        # outputs['pred_translation'] -> (bd, n_queries, 3);
        # outputs['pred_rotation']    -> (bs, n_queries, 3, 3)
        # outputs['pred_boxes']       -> (bs, n_queries, 4) -> (cx, cy, w, h) normalized
        start_ = time.time()
        outputs, n_boxes_per_sample = self.model(frame, None)
        poet_time = time.time() - start_

        frame_str = str(f"frame{frame_number}")
        self.results[frame_str] = {}

        if outputs is None:
            logger.warn("No prediction ... 'outputs' is None!")
            return None, None, None, None, None

        val = False
        for i in outputs["pred_boxes"].detach().cpu().tolist():
            for a in i:
                for b in a:
                    if b != -1:
                        val = True

        if val == False:
            logger.warn("No prediction ...")
            return None, None, None, None, None

        # TODO: Check if n_boxes_per_sample[0] is non-zero if nothing predicted
        # Iterate over all the detected predictions
        result: dict = {}
        for d in range(n_boxes_per_sample[0]):
            pred_t = np.array(outputs['pred_translation'][0][d].detach().cpu().tolist())
            pred_rot = np.array(outputs['pred_rotation'][0][d].detach().cpu().tolist())
            pred_box = np.array(outputs['pred_boxes'][0][d].detach().cpu().tolist())
            pred_class = int(outputs['pred_classes'][0][d].detach().cpu())

            #########################
            # TODO: Refactor object!!
            R, t = self.transform_to_cam(pred_rot, pred_t, "cabinet")

            img = None
            if self.draw:
                # Draw predicted bounding box
                detections = sv.Detections(
                    xyxy=np.array([self.transform_bbox(pred_box, (w, h))]),
                    class_id=np.array([0]))  # transform normalized cxcywh to xyxy
                box_annotator = sv.BoundingBoxAnnotator()
                # img = Image.fromarray(frame_orig)
                img = frame_orig
                annotated_frame = box_annotator.annotate(scene=img, detections=detections)
                img = np.array(annotated_frame)

            result[d] = {
                "t": t,
                "rot": R,
                "box": pred_box,  # format: cxcywh
                "class": pred_class,
                "img": img,
            }

        self.results[frame_str] = result
        if not result: return None, time.time() - start_, poet_time, None, None

        t = mean_translation(result)
        R = mean_rotation(result)

        t_rmse = self.rmse_translation(t, gt.t.data())
        R_rmse = self.rmse_rotation(R, gt.R.data())

        # if verbose:
        #     txt = f"[{frame_number:04d}] Total: {time.time() - start_:.4f}s | PoEt: {poet_time:>.4f}s | x: {t[0]:.4f} | y: {t[1]:.4f} | z : {t[2]:.4f} | RMSE (t): {t_rmse:.4f}, (R): {R_rmse:.4f}"
        #     logger.succ(txt)
        #     log_file.write(txt + "\n")

        return result, time.time() - start_, poet_time, t_rmse, R_rmse