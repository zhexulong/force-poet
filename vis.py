import cv2
import numpy as np
import json


def visualize_bop_box(data_path, scene_id, im_id, box_format="xywh"):
  """
  This function visualizes a bounding box from a BOP detection result file,
  projecting the corners based on the rotation matrix and camera intrinsics.

  Args:
      data_path (str): Path to the BOP dataset directory containing images.
      scene_id (int): Scene ID from the detection result.
      im_id (int): Image ID from the detection result.
      box_format (str, optional): Format of the bounding box coordinates in the result file.
          Defaults to "xywh" (x-min, y-min, width, height). Other options include "xyxy" (x-min, y-min, x-max, y-max).
  """
  # Load the image based on scene and image IDs
  image_path = f"{data_path}/test/{scene_id}/images/{im_id:06d}.jpg"
  image = cv2.imread(image_path)

  # Read detection results from the file
  with open("/media/sebastian/TEMP/poet/results/gt_ycbv-test.csv", "r") as f:
    next(f)
    lines = f.readlines()
  
  # Find the specific detection line for this image and object
  for line in lines:
    data = line.strip().split(",")
    if int(data[0]) == scene_id and int(data[1]) == im_id:
      # Extract bounding box coordinates based on format
      if box_format == "xywh":
        x_min, y_min, width, height = [float(val) for val in data[4].split(" ")]
      elif box_format == "xyxy":
        x_min, y_min, x_max, y_max = [float(val) for val in data[4].split(" ")]
        width = x_max - x_min
        height = y_max - y_min
      else:
        raise ValueError("Unsupported box format. Choose between 'xywh' or 'xyxy'")

      # Define 8 corners of a 3D bounding box (assuming a cuboid) with center at (x_min + width/2, y_min + height/2)
      box_corners_3d = np.array([
          [-width/2, -height/2,  1],
          [ width/2, -height/2,  1],
          [ width/2,  height/2,  1],
          [-width/2,  height/2,  1],
          [-width/2, -height/2, -1],
          [ width/2, -height/2, -1],
          [ width/2,  height/2, -1],
          [-width/2,  height/2, -1],
      ])

      # Extract rotation matrix from the data (assuming it's after commas following the bounding box coordinates)
      rotation_matrix = np.array([[float(val) for val in row.split(" ")] for row in data[5:8]])

      # Load camera intrinsic parameters from scene_camera.json
      camera_path = f"{data_path}/test/{scene_id}/scene_camera.json"
      with open(camera_path, "r") as f:
          camera_data = json.load(f)
      camera_matrix = np.array(camera_data["cam_K"])  # Assuming "cam_K" key stores the intrinsic matrix

      # Project bounding box corners onto the image plane
      projection_matrix = np.concatenate((camera_matrix, np.array([[0], [0], [1]])), axis=1)  # Assuming no distortion
      projected_corners = np.matmul(projection_matrix, np.matmul(rotation_matrix, box_corners_3d.T).T)

      # Normalize projected points (might be necessary depending on the projection matrix)
      projected_corners = projected_corners[:, :2] / projected_corners[:, 2]  # Normalize by z-component

      # Extract projected x and y coordinates
      projected_x = projected_corners[:, 0]
      projected_y = projected_corners[:, 1]

      # Find minimum and maximum projected points for drawing the box
      min_x = int(np.min(projected_x))
      max_x = int(np.max(projected_x))
      min_y = int(np.min(projected_y))
      max_y = int(np.max(projected_y))

      # Draw the projected bounding box on the image
      cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)  # Green for visualization

      # Display the image with the bounding box
      cv2.imshow(f"BOP Detection - Scene {scene_id}, Image {im_id}", image)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
      break  # Exit after finding the specific detection

# Edit these values with your data paths and desired box format
data_path = "/media/sebastian/TEMP/poet/datasets/ycbv"
box_format = "xywh"  # Change to "xyxy" if your results use that format

# visualize_bop_box(data_path, scene_id, im_id, box_format)


# Read detection results from the file
# with open("/media/sebastian/TEMP/poet/results/backbone_ycbv-test.csv", "r") as f:
with open("./output/bop_gt/ycbv.csv", "r") as f:
  next(f)
  lines = f.readlines()

for line in lines:
  data = line.strip().split(",")
  scene_id = int(data[0])
  img_id = int(data[1])
  obj_id = int(data[2])
  score = float(data[3])
  R_obj = data[4]
  t_obj = data[5]
  time = float(data[6])


  R_values = list(map(float, R_obj.split()))
  R_obj = np.array(R_values).reshape(3, 3)

  t_values = list(map(float, t_obj.split()))
  t_obj = np.array(t_values)

  # Load camera intrinsic parameters from scene_camera.json
  camera_path = f"{data_path}/test_all/test/{scene_id:06d}/scene_camera.json"
  with open(camera_path, "r") as f:
      camera_data = json.load(f)
  cam_intr = camera_data.get(str(img_id))

  cam_K = cam_intr.get("cam_K")
  cam_R = cam_intr.get("cam_R_w2c")
  cam_t = cam_intr.get("cam_t_w2c")

  cam_K = np.array(cam_K).reshape(3, 3)
  cam_R = np.array(cam_R).reshape(3, 3)
  cam_t = np.array(cam_t)

  tmp = np.hstack((cam_R, cam_t.reshape(3, 1)))

  #C = np.dot(cam_K, tmp)

  # t_obj = np.append(t_obj, 1)

  #t_img = np.dot(cam_K, np.dot(tmp, t_obj))
  t_img = np.dot(cam_K, t_obj)
  t_img = t_img / t_img[2]

  # Load the image based on scene and image IDs
  image_path = f"{data_path}/test_all/test/{scene_id:06d}/rgb/{img_id:06d}.png"
  image = cv2.imread(image_path)

  x = int(t_img[0])
  y = int(t_img[1])

  radius = 5
  color = (0, 255, 0)  # Red color (BGR format)
  thickness = -1  # Filled circle
  cv2.circle(image, (x, y), radius, color, thickness)


  cv2.imshow(f"BOP Detection - Scene {scene_id:06d}, Image {img_id:06d}", image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

