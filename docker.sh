
# Start docker with bash
sudo docker run --rm --gpus all -it aaucns/poet:latest bash


docker run --entrypoint= -v /home/sebastian/repos/poet:/opt/project -v /media/sebastian/TEMP/poet/datasets/ycbv:/data -v /media/sebastian/TEMP/poet/output:/output --rm --gpus all aaucns/poet:latest bash