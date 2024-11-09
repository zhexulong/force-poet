import json
import re
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from dataset.common import dimensions
from tabulate import tabulate

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

def zoom_factory(ax, base_scale=2.):
    def zoom_fun(event):
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0]) * .5
        cur_yrange = (cur_ylim[1] - cur_ylim[0]) * .5
        xdata = event.xdata  # get event x location
        ydata = event.ydata  # get event y location
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
            print
            event.button
        # set new limits
        ax.set_xlim([xdata - cur_xrange * scale_factor,
                     xdata + cur_xrange * scale_factor])
        ax.set_ylim([ydata - cur_yrange * scale_factor,
                     ydata + cur_yrange * scale_factor])
        plt.draw()  # force re-draw

    fig = ax.get_figure()  # get the figure of interest
    # attach the call back
    fig.canvas.mpl_connect('scroll_event', zoom_fun)

    # return the function
    return zoom_fun

def load_data(file_path):
    """Load JSON data from a file and convert to numpy arrays."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    frames = {}
    for frame_key, frame_value in data.items():
        frames[frame_key] = {}
        for object_key, object_value in frame_value.items():
            frames[frame_key][object_key] = {
                't': np.array(object_value['t']),
                'rot': np.array(object_value['rot']),
                'box':  np.array(object_value['box']) if 'box' in object_value else np.array(object_value['bbox']),
                'class': np.array(object_value['class'])
            }
    
    return frames

def position_error(pos1, pos2):
    """Calculate the Euclidean distance (L2 norm) between two arrays."""
    return np.sqrt(np.sum((pos1 - pos2) ** 2))

def rotation_error(rot1, rot2) -> float:
    """Calculate the geodesic distance on the rotation manifold of a predicted (rot1) and ground truth (rot2) rotation"""
    rot = np.matmul(rot1, rot2.T)
    trace = np.trace(rot)
    if trace < -1.0:
        trace = -1
    elif trace > 3.0:
        trace = 3.0
    angle_diff = np.degrees(np.arccos(0.5 * (trace - 1)))
    return angle_diff


def rmse_on_percentile(errors: np.ndarray, percentile: int) -> float:
    threshold = np.percentile(errors, percentile)
    filtered = errors[errors <= threshold]
    return np.sqrt(np.mean(filtered ** 2))


def transform_to_cam(R, t):
    # t = bottom_object_origin(object=obj, t=t)
    return R.T, np.dot(-R.T, t)

obj = "cabinet_1"

# pred = load_data(f'/home/sebastian/repos/poet/results/inf/dino_yolo_custom/results.json')
pred = load_data(f'results/{obj}/results.json')
# gt_obj = load_data(f'./dataset/records/{obj}/gt_obj.json')
# gt_cam = load_data(f'./dataset/records/{obj}/gt_cam.json')
gt_obj = load_data(f'./dataset/records/{obj}/gt_obj.json')
gt_cam = load_data(f'./dataset/records/{obj}/gt_cam.json')

obj = re.sub(r'_\d+$', '', obj)

# 3D plot
fig = plt.figure(figsize=(24, 24))

ax = fig.add_subplot(projection="3d")
ax.set_aspect('equal')
zoom_factory(ax)

# 3D bbox of object
w = dimensions[obj]["w"]  # x
d = dimensions[obj]["d"]  # y
h = dimensions[obj]["h"]  # z

half_w = w / 2
half_d = d / 2
half_h = h / 2

# Define the vertices of the cuboid
vertices = np.array([
    [-half_w, -half_d, 0],  # 0: Bottom face
    [ half_w, -half_d, 0],  # 1: Bottom face
    [ half_w,  half_d, 0],  # 2: Bottom face
    [-half_w,  half_d, 0],  # 3: Bottom face
    [-half_w, -half_d, h],  # 4: Top face
    [ half_w, -half_d, h],  # 5: Top face
    [ half_w,  half_d, h],  # 6: Top face
    [-half_w,  half_d, h]   # 7: Top face
])

# Define faces of the bounding box
faces = [
    [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom face
    [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top face
    [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front face
    [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right face
    [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back face
    [vertices[3], vertices[0], vertices[4], vertices[7]]   # Left face
]

bounding_box = Poly3DCollection(faces, facecolors='g', edgecolors='g', alpha=0.3)
ax.add_collection3d(bounding_box)
ax.quiver(0, 0, h/2, 1, 0, 0, color='r', length=0.1)
ax.quiver(0, 0, h/2, 0, 1, 0, color='g', length=0.1)
ax.quiver(0, 0, h/2, 0, 0, 1, color='b', length=0.1)

# Visualize floor
x_plane = np.linspace(-1.5, 1.5, 10)
y_plane = np.linspace(-1.5, 1.5, 10)
x_plane, y_plane = np.meshgrid(x_plane, y_plane)
z_plane = np.zeros_like(x_plane)  # Plane at z = 0
ax.plot_surface(x_plane, y_plane, z_plane, color='k', alpha=0.2)

gt_x_traj = []
gt_y_traj = []
gt_z_traj = []

pred_x_traj = []
pred_y_traj = []
pred_z_traj = []

gt_positions = np.empty((0, 3))
pred_positions = np.empty((0, 3))

gt_rotations = np.empty((0, 3, 3))
pred_rotations = np.empty((0, 3, 3))

total_imgs = 0
for frame_key in pred:
    for object_key in pred[frame_key]:
        if object_key not in gt_cam[frame_key]: continue
        gt_pos = gt_cam[frame_key][object_key]['t']
        gt_rot = gt_cam[frame_key][object_key]['rot']

        pred_rot, pred_pos = transform_to_cam(pred[frame_key][object_key]['rot'], pred[frame_key][object_key]['t'])

        gt_positions = np.vstack([gt_positions, gt_pos])
        pred_positions = np.vstack([pred_positions, pred_pos])

        gt_rotations = np.vstack([gt_rotations, [gt_rot]])
        pred_rotations = np.vstack([pred_rotations, [pred_rot]])
    total_imgs += 1


t_err = np.empty(0)
rot_err = np.empty(0)
for gt_pos, pred_pos, gt_rot, pred_rot in zip(gt_positions, pred_positions, gt_rotations, pred_rotations):
    t_err = np.append(t_err, position_error(gt_pos, pred_pos))
    rot_err = np.append(rot_err, rotation_error(gt_rot, pred_rot))

# Visualize ground truth poses
ax.scatter(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], s=64, c='b', marker='o')
ax.quiver(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], gt_rotations[:, 0, 0], gt_rotations[:, 1, 0], gt_rotations[:, 2, 0], color='b', length=0.1)

# Visualize predicted poses
ax.scatter(pred_positions[:, 0], pred_positions[:, 1], pred_positions[:, 2], s=64, c='r', marker='o')
ax.quiver(pred_positions[:, 0], pred_positions[:, 1], pred_positions[:, 2], pred_rotations[:, 0, 0], pred_rotations[:, 1, 0], pred_rotations[:, 2, 0], color='r', length=0.1)

# Visualize L2 position error
for gt_pos, pred_pos, err in zip(gt_positions, pred_positions, t_err):
    ax.plot([gt_pos[0], pred_pos[0]], [gt_pos[1], pred_pos[1]], [gt_pos[2], pred_pos[2]], color="gray", linestyle="dashed", alpha=0.5)
    midpoint = (gt_pos + pred_pos) / 2
    ax.text(midpoint[0], midpoint[1], midpoint[2], f'{err:.2f}', color='black', fontsize=16)

# Calculate overall RMSE for translation and rotation
translation_rmse = rmse_on_percentile(t_err, 75)
rotation_rmse = rmse_on_percentile(rot_err, 75)

headers = ["Metric", "RMSE", "RMSE (75th)", "RMSE (50th)"]

data = [
    ["Translation", f"{np.mean(t_err):.4f} m", f"{rmse_on_percentile(t_err, 75):.4f} m", f"{rmse_on_percentile(t_err, 50):.4f} m"],
    ["Rotation", f"{np.mean(rot_err):.4f} °", f"{rmse_on_percentile(rot_err, 75):.4f} °", f"{rmse_on_percentile(rot_err, 50):.4f} °"],
]

print(tabulate(data, headers=headers, tablefmt="pretty"))
print(f"Processed images: {total_imgs}")

# Plot the trajectory of the ground truth positions as a continuous line
# ax.plot(gt_x_traj, gt_y_traj, gt_z_traj, color='b', linestyle='-', linewidth=2, label='GT Trajectory')
# ax.plot(pred_x_traj, pred_y_traj, pred_z_traj, color='r', linestyle='-', linewidth=2, label='PD Trajectory')

# ax.set_zlim(0, 3)
ax.set_aspect("equal", adjustable="datalim")

# Set the plot view 
ax.view_init(elev=20, azim=45)
ax.dist = 8

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Trajectory visualization
plt.figure(figsize=(14, 16))
indices = [i for i in range(len(gt_positions))]

# Subplot 1: X-axis trajectory
plt.subplot(3, 1, 1)  # 3 rows, 1 column, plot 1
plt.plot(indices, gt_positions[:, 0], color="b", label="Ground truth")
plt.plot(indices, pred_positions[:, 0], color="r", label="Predicted")
plt.grid(True)
plt.xlabel("Time")
plt.ylabel("X (Meters)")
plt.title("X - Trajectory")
plt.legend()

# Subplot 2: Y-axis trajectory
plt.subplot(3, 1, 2)  # 3 rows, 1 column, plot 2
plt.plot(indices, gt_positions[:, 1], color="b", label="Ground truth")
plt.plot(indices, pred_positions[:, 1], color="r", label="Predicted")
plt.grid(True)
plt.xlabel("Time")
plt.ylabel("Y (Meters)")
plt.title("Y - Trajectory")
plt.legend()

# Subplot 3: Z-axis trajectory
plt.subplot(3, 1, 3)  # 3 rows, 1 column, plot 3
plt.plot(indices, gt_positions[:, 2], color="b", label="Ground truth")
plt.plot(indices, pred_positions[:, 2], color="r", label="Predicted")
plt.grid(True)
plt.xlabel("Time")
plt.ylabel("Z (Meters)")
plt.title("Z - Trajectory")
plt.legend()

# Adjust layout to prevent overlap
plt.tight_layout()

plt.show()
