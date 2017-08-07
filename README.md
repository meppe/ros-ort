# ros-ort -- Object Detection and Tracking for ROS.

Note that this is work in progress. It should work when you follow the instructions below but that is not guaranteed. This work is based on py-faster-rcnn, which is available here: https://github.com/rbgirshick/py-faster-rcnn

This repo contains start scripts, and code to run dockerized object recognition and tracking based on ROS. Docker images are based on Ubuntu 16.04 and ROS kinetic. Object detection is based on FRCNN. Scripts have been tested on Ubuntu 14.04 and 16.04.

The system consists of several ROS nodes that each run as a separate docker container. When running for the first time, docker will download the images which can be quite big, so this will take a while. Before running, make sure that you have installed Docker (on Ubuntu 16.04 do ```sudo apt-get install docker```) and nvidia-docker as described here: https://github.com/NVIDIA/nvidia-docker. To be able to run Docker without superuse privileges, you must also add your user to the docker group, as described here: http://askubuntu.com/questions/477551/how-can-i-use-docker-without-sudo. For Ubuntu 14.04 it is recommended to not use the docker that comes with Ubuntu, but install docker from here: https://docs.docker.com/engine/installation/linux/ubuntulinux/

For GPU support you need an NVIDIA-driver version >=8.0 and a GPU.

Once you have met the above extrendencies, clone this repository and initialize the submodules using ```git submodule update --init --recursive``. 

## Quickstart - Object recognition and tracking with pretrained networks

To see a first quick demo, follow the following steps:

1. Download caffe network models. 
	To download the pretrained network models, run ```./download_caffemodels.sh```.

2. Run ros nodes. You can run all required nodes automatically by running ```./run_nico_detection.sh```. This script essentially starts the following scripts below:

	1. ROS core -- The ROS core node is started by running the script ```./run_roscore.sh```

	2. Video stream -- 
		Starts the video stream . Started by ```run_video_stream.sh```. By default, it reads from ``/dev/video0`` and publishes `/frcnn_input/camera_raw`. Inspect the script to understand streaming from a video file or streaming from another video source. 
	
	3. Video preview -- 
		This is optional and opens a window that shows the original input video stream. It is started with the script ```./run_video_view.sh```. It will subscribe to the topic frcnn_input/camera_raw. 
		
	4. Object detection -- 
		This runs the Faster-RCNN-based object detection. It is started by running the script ```./run_frcnn_detect.sh``` for CPU use or ```./run_frcnn_detect.sh --gpu``` for running it on your GPU. It subscribes to the topic `/frcnn_input/camera_raw`, and it will publish the topic `/frcnn/bb` for the bounding box information and the topic `/frcnn/bb_img` to re-publish those video frames from the input stream which haven been processed. The frequency depends on the processing speed. Without a GPU around 0.1 -- 0.2 Hz, and with a NVIDIA Titan GPU around 5 Hz. 
		
	5. Object tracking -- Started via ```run_frcnn_track.sh```. It clusters the bounding boxes delivered by the detector (subscribes to topics `/frcnn/bb` and `/frcnn/bb_img`) and assigns labels. The results are published on topic /frcnn/bb_img_tracking. 
		
		

	6. Tracking preview -- 
		This opens a window to show the tracking result. It subscribes to topic `/frcnn/bb_img_tracking`. It is started with ```run_video_tracking_view.sh```

	7. Text interface -- 
		The script ```./run_txt_interface.sh```  opens the text interface, where you can modify where image files are stored, which objects are to be detected, and whether masking is applied. It should be self-explanatory. 

## Interfacing with the object detection 

There are two ways to access the object detection data. If you just want a quick solution to use the system, you should use the temporary files generated by the tracker. By default, there are two files generated. Each frame overwrites these files. One file (default: "/tmp/img.jpg") is a .jpg image that includes the bounding boxes and also the attention focus if you set it accordingly in the text interface. The second file describes the bounding boxes along with confidence scores and class labels. The defaule path is "/tmp/detectiopns.txt". The file names can be modified with the text interface. The more solid and profound way is via ROS topics, which you can infer by inspecting the code in `tracker.py`. 
		
## Training own datasets

To train your own dataset follow the following steps:

1. Generate a sequence of frames from a video. This can be done by using the ``generate_frames.sh`` script. Please have a look at it to understand how it works. You have to specify a source folder as a mandatory argument to the script, which contains a set of videos. The target root data folder is set as variable `data_root` in the script; this is the location where the frames are stored. For each source video, a subfolder containing the individual frames is created in the target root folder. 

2. Annotate the frames using the tracker. Do so by starting the `run_annotate.sh` script (see known issues section below). Before doing that, make sure that your `DATA_ROOT` folder is set correctly in the `src/frcnn/scripts/run_annotate.py` file. The data folder is arranged according to the PASCAL_VOC convention. Specifically, you need to make sure that the following files are in place: 
	
	a. ``DATA_ROOT/<dataset_name>/<dataset_subset>/Annotations/<number>.xml`` (the annotation files, one for each frame)

	b. ``DATA_ROOT/<dataset_name>/<dataset_subset>/JPGImages/<number>.jpg`` (the image files, one for each frame)

	c. ``DATA_ROOT/<dataset_name>/<dataset_subset>/ImageSets/Main/train.txt, test.txt, val.txt, trainval.txt`` (files that define how the train test val spilt is done)

In the example case, ``<dataset_name>/<dataset_subset>`` is ``nico2017/nico2017``

See http://host.robots.ox.ac.uk/pascal/VOC/ for more details about the file structure.

3. Start the `run_training.sh` script. Take care that the datafolder is set appropriately, in both the python call via ``--set DATA_DIR /storage/data`` and also the docker volume mount for the data folder, in my case ``-v /storage/data:/storage/data``. The script reads data according to the VOC convention. 

## Score threshold values

Currently, the preferred way to set the threshold values is via command line options in the individual .sh files for each ROS node. The most important ones are 
	
	a. the ``--threshold`` in ``run_detect.sh``. This sets the threshold for the object detection

	b. the ``--class_threshold`` in ``run_track.sh``. This sets the cumulative class score for each class in each bounding box. Cumulative means accumulated score over time with a small decay. See ``tracker.py`` for details. 

	c. the ``--cum_threshold`` in ``run_trach.sh``. This sets the cumulative total score for each bounding box. Cumulative means accumulated sum of scores of detected classes over time, with a small decay. See ``tracker.py`` for details. 


## Known issues

1. With the docker version 17.03.1-ce (and probably neighbouring versions) you can not run ROS-ORT from an NFS drive due to volume mounting issues. Hence, if you are in a university network where your standard folder is on a network drive, you need to make sure you run the scripts from some local folder. 

2. For some reason, the `run_annotate.py` only works from within my pycharm ide. When drawing a bounding box after starting it from console using `run_annotate.sh`, it crashes. It crashes at line 219 in `annotator.py`, i.e., at `key = cv2.waitKey(1) & 0xFF`. There are these error messages when starting the script:
```
libGL error: No matching fbConfigs or visuals found
libGL error: failed to load driver: swrast
```

I have no idea why this is the case...




