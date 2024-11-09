from scipy.spatial.transform import Rotation
import numpy as np
import os

def getPosition(content):
    return [float(content[1]), float(content[2]), float(content[3])]

def getRotation(content):
    return [float(content[4]), float(content[5]), float(content[6]), float(content[7])]

def correct_pose_matrix(position, rotation_qat):
    rotation_matrix = Rotation.from_quat(rotation_qat)
    rotation_90_z = Rotation.from_euler('z', 90, degrees=True)

    corrected_position = rotation_90_z.apply(position)
    corrected_rotation = rotation_90_z * rotation_matrix
    return corrected_position, corrected_rotation.as_quat()

base_path = f"./records_23_09_24"
cnt = 0
for object in os.listdir(base_path):
    print(f"[INFO] Processing '{object}' ...")

    if not os.path.isfile(f"{base_path}/{object}/gt_cam.txt"):
        print(f"[FAIL] gt_cam.txt doesn't exist for {object}! Skipping...")
        print("--------------------------------------------------")
        continue

    if os.path.isfile(f"{base_path}/{object}/gt_cam_old.txt"):
        print(f"[FAIL] gt_cam_old.txt already exists! for {object}! Skipping...")
        print("--------------------------------------------------")
        continue

    os.rename(f"{base_path}/{object}/gt_cam.txt", f"{base_path}/{object}/gt_cam_old.txt")

    # This is the raw file of the Drone's ground truth pose recorded the optitrack system.
    with open(f"{base_path}/{object}/gt_cam_old.txt", "r") as file:
        f_in = [line.rstrip() for line in file]

    f_out = open(f"{base_path}/{object}/gt_cam.txt", "w")
    
    for line in f_in:
        content = line.split(" ")

        frame = content[0]
        position = np.array(getPosition(content))
        rotation = getRotation(content)

        position, rotation = correct_pose_matrix(position, rotation)
         # Format the corrected output as a string
        corrected_line = f"{frame} {position[0]} {position[1]} {position[2]} {rotation[0]} {rotation[1]} {rotation[2]} {rotation[3]}\n"
        f_out.write(corrected_line)

    cnt += 1

print(f"Successfully corrected gt information for {cnt} records!")



