# Wrapper for Scaled-YOLOv4
This is repository provides a wrapper for [WongKinYiu's Scaled-YOLOv4 implementation](https://github.com/WongKinYiu/ScaledYOLOv4/tree/yolov4-csp) in PyTorch. In contrast to the original implementation, the wrapper returns more information during a forward pass. Check out the original repository for more detailed information.

## Installation
```sh
# create the docker container, you can change the share memory size if you have more.
nvidia-docker run --name yolov4_csp -it -v your_coco_path/:/coco/ -v your_code_path/:/yolo --shm-size=64g nvcr.io/nvidia/pytorch:20.11-py3

# apt install required packages
apt update
apt install -y zip htop screen libgl1-mesa-glx

# pip install required packages
pip install seaborn thop

# install mish-cuda if you want to use mish activation
# https://github.com/thomasbrandon/mish-cuda
# https://github.com/JunnYu/mish-cuda
cd /
git clone https://github.com/JunnYu/mish-cuda
cd mish-cuda
python setup.py build install

# go to code folder
cd /yolo
```

