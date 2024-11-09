import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import message_filters
from geometry_msgs.msg import PoseStamped, Twist
import yaml
from std_msgs.msg import Empty
import numpy as np
import math
import os
#import ffmpegcv
from djitellopy import Tello
from PIL import Image as im
from time import sleep
import time
import ffmpegcv
import threading

base_path = "dataset/records"

id = 0
files = [f for f in os.listdir(base_path) if os.path.isfile(f)]
for file in files:
  print(file)
  if "avi" in file:
    id = id + 1

if not os.path.exists(f"{base_path}/{id}/imgs/"):
  os.makedirs(f"{base_path}/{id}/imgs/")

counter = 0
frame_number = 0

tello = Tello()
tello.connect()

true_pub = rospy.Publisher("true_pose", PoseStamped, queue_size=100)

tello.streamon()

frame_read = tello.get_frame_read()

def store_pose(pose):
  global frame_number

  name = "frame" + str(frame_number) + ".png"
  print(name)
  seq_dir = f"{base_path}/{id}/"
  print(seq_dir)

  line = name + " " + str(pose.pose.position.x) + " " + str(pose.pose.position.y) + " " + str(pose.pose.position.z) + " " + str(pose.pose.orientation.x)+ " " + str(pose.pose.orientation.y) + " " + str(pose.pose.orientation.z) + " " + str(pose.pose.orientation.w) + "\n"

  out_file = open(os.path.join(seq_dir, f"gt_cam.txt"), "a")
  out_file.write(line)




def second_thread():
  global frame_read
  out = ffmpegcv.VideoWriter(f'{base_path}/{id}/video.avi', None, 20, resize=(960, 720), pix_fmt='rgb24')

  while not rospy.is_shutdown():
    frame = frame_read.frame
    if frame.shape != (720,960,3):
      continue
    out.write(frame)
    time.sleep(1/30)


def callback(pose):
  global counter
  global frame_number
  global frame_read
  counter += 1

  if counter % 10 == 0:
    print("received image: ", counter)

    print("x: ", pose.pose.position.x)
    store_pose(pose)
    path = f"{base_path}/{id}/imgs/"
    name = "frame" + str(frame_number) + ".png"

    cv2.imwrite(os.path.join(path, name), cv2.cvtColor(frame_read.frame, cv2.COLOR_RGB2BGR))

    frame_number += 1

    pose.header.frame_id = "world"

    true_pub.publish(pose)


def main():

  sleep(5)

  rospy.init_node('image_saver')

  vid_thread = threading.Thread(target=second_thread)
  vid_thread.start()

  pose_topic = "/mocap_node/tello/pose"

  pose_sub = rospy.Subscriber(pose_topic, PoseStamped, callback=callback)

  rospy.spin()

if __name__ == '__main__':
  main()
  tello.streamoff()