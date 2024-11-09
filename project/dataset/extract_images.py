import cv2
import os

base_path = "records/"

cnt = 0
for object in os.listdir(base_path):
  print(f"Processing '{object}'...")
  if not os.path.exists(f"{base_path}/{object}/imgs"):
      os.makedirs(f"{base_path}/{object}/rgb")
      print(f"[INFO] Directory for '{object}' created successfully!")
  else:
      print(f"[FAIL] Directory 'rgb' for '{object}' already exists! Skipping record...")
      print("---------------------------------------")
      continue

  vidcap = cv2.VideoCapture(f"{base_path}/{object}/video.avi")
  if not vidcap.isOpened():
     print(f"[FAIL] Couldn't open video for {object}! Skipping record...")
     print("---------------------------------------")
     continue

  success,image = vidcap.read()
  count = 0
  while success:
    image = cv2.resize(image, (640, 480))
    cv2.imwrite(f"{base_path}/{object}/imgs/frame%d.png" % count, image)     # save frame as JPEG file      
    success,image = vidcap.read()
    count += 1

  cnt = cnt + 1
  print(f"[SUCC] Written {count} frames for '{object}'!")
  print("---------------------------------------")

print(f"Successfully extracted {cnt} recorded video!")