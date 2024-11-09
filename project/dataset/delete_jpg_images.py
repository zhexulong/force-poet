import os

def delete_jpg_images(folder_path):
    try:
        # List all files in the specified directory
        files = os.listdir(folder_path)
        
        # Iterate through all files
        for file in files:
            # Check if the file ends with '.jpg'
            if file.lower().endswith('.jpg'):
                file_path = os.path.join(folder_path, file)
                # Delete the file
                os.remove(file_path)
                print(f"Deleted: {file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

objects = ["book", "book1", "cabinet", "cabinet1", "can", "can1", "chair", "chair1","chair2", "doll", "doll1", "table"]
for object in objects:
    # Specify the folder path
    folder_path = f'files/{object}/rgb'

    # Call the function to delete jpg images
    delete_jpg_images(folder_path)