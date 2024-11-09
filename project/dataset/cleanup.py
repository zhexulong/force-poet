import os

def process_gt_cam_file(folder_path):
    # Paths to the required files and folders
    gt_cam_path = os.path.join(folder_path, 'gt_cam.txt')
    gt_cam_old_path = os.path.join(folder_path, 'gt_cam_old.txt')
    imgs_folder_path = os.path.join(folder_path, 'imgs')

    # Check if the required paths exist
    if not os.path.exists(gt_cam_path) or not os.path.exists(imgs_folder_path):
        print(f"Skipping folder {folder_path}, missing gt_cam.txt or imgs folder.")
        return

    # Read the content of the original gt_cam.txt
    with open(gt_cam_path, 'r') as file:
        gt_cam_lines = file.readlines()

    # Prepare the list of image names present in the imgs folder
    imgs_files = set(os.listdir(imgs_folder_path))

    if (len(imgs_files) != len(gt_cam_lines)):
        print("Image count and lines in gt file missmatch!!!")

    # Filter the gt_cam.txt content based on the presence of corresponding images
    not_present = []
    filtered_lines = []
    for line in gt_cam_lines:
        img_name = line.split()[0]
        if img_name in imgs_files:
            filtered_lines.append(line)
        else:
            not_present.append(img_name)

    # Rename the old gt_cam.txt to gt_cam_old.txt
    # os.rename(gt_cam_path, gt_cam_old_path)

    # Write the filtered content back to gt_cam.txt
    with open(gt_cam_path, 'w') as file:
        file.writelines(filtered_lines)

    print(f"Processed {folder_path}: {len(filtered_lines)} valid entries saved and {len(not_present)} removed of {len(imgs_files)} total.")


def process_all_folders(root_folder):
    # Iterate through all subfolders in the root folder
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            process_gt_cam_file(folder_path)


# Example usage:
root_folder = 'records_12_09_24/'  # Replace this with the actual path to your root folder
process_all_folders(root_folder)
