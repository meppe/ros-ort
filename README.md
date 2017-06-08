# ros-ort -- Object Detection and Tracking for ROS.

Note that this is work in progress. It should work when you follow the instructions below but that is not guaranteed. This work is based on py-faster-rcnn, which is available here: https://github.com/rbgirshick/py-faster-rcnn

This repo contains start scripts, and code to run dockerized object recognition and tracking based on ROS. Docker images are based on Ubuntu 16.04 and ROS kinetic. Object detection is based on FRCNN. Scripts have been tested on Ubuntu 14.04 and 16.04.

The system consists of several ROS nodes that each run as a separate docker container. When running for the first time, docker will download the images which can be quite big, so this will take a while. Before running, make sure that you have installed Docker (on Ubuntu 16.04 do ```sudo apt-get install docker```) and nvidia-docker as described here: https://github.com/NVIDIA/nvidia-docker. To be able to run Docker without superuse privileges, you must also add your user to the docker group, as described here: http://askubuntu.com/questions/477551/how-can-i-use-docker-without-sudo. For Ubuntu 14.04 it is recommended to not use the docker that comes with Ubunut, but install docker from here: https://docs.docker.com/engine/installation/linux/ubuntulinux/

For GPU support you need an NVIDIA-driver version >=8.0 and a GPU.

Once you have met the above extrendencies, clone this repository and initialize the submodules using ```git submodule update --init --recursive``. 

## Quickstart - Object recognition and tracking with pretrained networks

To see a first quick demo, follow the following steps:

1. ROS core
	The ROS core node. Start by running the script ```./run_roscore.sh```

2. Video preview
	This is optional and opens a window that shows the original input video stream. Start by running the script ```./run_video_view.sh```. It will subscribe to the topic frcnn_input/camera_raw. 
	
3. Object detection
	This runs the Faster-RCNN-based object detection. Start by running the script ```./run_frcnn_detect.sh``` for CPU use or ```./run_frcnn_detect.sh --gpu``` for running it on your GPU. It will subscribe to the topic /frcnn_input/camera_raw, and it will publish the topic /frcnn/bb for the bounding box information and the topic /frcnn/bb_img to re-publish those video frames from the input stream which haven been processed. The frequency depends on the processing speed. Without a GPU around 0.1 -- 0.2 Hz.
	
4. Object tracking
	This runs the tracking. It clusters the bounding boxes delivered by the detector (subscribes to topics /frcnn/bb and /frcnn/bb_img) and assigns labels. The results are published on topic /frcnn/bb_img_tracking. 
	To run the node do ```run_frcnn_track.sh```.
	
5. Tracking preview
	This opens a window to show the tracking result. It subscribes to topic /frcnn/bb_img_tracking. To run the node do ```run_video_stream.sh```
	
6. Video stream
	Once all nodes are running start the video stream. To do this, run `./run_video_stream`.

## Training own datasets

To train your own dataset follow the following steps:

1. Generate a sequence of frames from a video. This can be done by using the ``generate_frames.sh`` script. Please have a look at it to understand how it works. You have to specify a source folder as a mandatory argument to the script, which contains a set of videos. The target root data folder is set as variable `data_root` in the script; this is the location where the frames are stored. For each source video, a subfolder containing the individual frames is created in the target root folder. 

2. Annotate the frames using the tracker. Do so by starting the `run_annotate.sh` script. Before doing that, make sure that your `DATA_ROOT` folder is set correctly in the `src/frcnn/scripts/run_annotate.py` file. The data folder is arranged according to the PASCAL_VOC convention. See http://host.robots.ox.ac.uk/pascal/VOC/ for details.

3. Start the `run_training.sh` script. Take care that the datafolder is set appropriately, in both the python call via ``--set DATA_DIR /storage/data`` and also the docker volume mount for the data folder, in my case ``-v /storage/data:/storage/data``. The script reads data according to the VOC convention. 

## Known issues
With the docker version 17.03.1-ce (and probably neighbouring versions) you can not run ROS-ORT from an NFS drive due to volume mounting issues

For some reason, the `run_annotate.py` only works from within my pycharm ide. When drawing a bounding box after starting it from console using `run_annotate.sh`, it crashes. There are these error messages:
```
libGL error: No matching fbConfigs or visuals found
libGL error: failed to load driver: swrast
```





