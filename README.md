# ros-ort

This repo contains start scripts, possible Dockerfiles and some code to run dockerized object recognition and tracking based on ROS. Docker images are based on Ubuntu 16.04 and ROS kinetic. Object detection is based on FRCNN. 

The system consists of several ROS nodes that each run as a separate docker container. When running for the first time, docker will download the images which can be quite big, so this will take a while. Before running, make sure that you have installed Docker (on Ubuntu ```sudo apt-get install docker```) and nvidia-docker as described here: https://github.com/NVIDIA/nvidia-docker. To be able to run Docker without superuse privileges, you must also add your user to the docker group, as described here: http://askubuntu.com/questions/477551/how-can-i-use-docker-without-sudo.

1. ROS core
	The ROS core node. Start by running the script ```./run_roscore.sh```

2. Video preview
	This is optional and opens a window that shows the original input video stream. Start by running the script ```./run_video_view.sh```. It will subscribe to the topic frcnn_input/camera_raw. 
	
3. Object detection
	This runs the Faster-RCNN-based object detection. Start by running the script ```./run_frcnn_detect.sh```. It will subscribe to the topic /frcnn_input/camera_raw, and it will publish the topic /frcnn/bb for the bounding box information and the topic /frcnn/bb_img to re-publish those video frames from the input stream which haven been processed. The frequency depends on the processing speed. Without a GPU around 0.1 -- 0.2 Hz.
	
4. Object tracking
	This runs the tracking. It clusters the bounding boxes delivered by the detector (subscribes to topics /frcnn/bb and /frcnn/bb_img) and assigns labels. The results are published on topic /frcnn/bb_img_tracking. 
	To run the node do ```run_frcnn_track.sh```.
	
5. Tracking preview
	This opens a window to show the tracking result. It subscribes to topic /frcnn/bb_img_tracking. To run the node do ```run_vide_stream.sh```
	
6. Video stream
	Once all nodes are running start the video stream. To do this, run ```./run_video_stream``.


