#!/bin/bash
docker rm ros_frcnn_training
echo "Note: Training must be run with nvidia-docker and GPU-support because some layers in caffe do not support cpu mode. Running nvidia-docker with GPU"
docker pull meppe78/ros-kinetic-frcnn-training
nvidia-docker run \
--rm \
-v /$(pwd)/src/frcnn:/opt/ros-ort/src/frcnn \
-v /$(pwd)/src/ort_msgs:/opt/ros-ort/src/ort_msgs \
-v /$(pwd)/output:/opt/ros-ort/output \
-v /$(pwd)/src/frcnn/src/py-faster-rcnn:/opt/ros-ort/src/frcnn/src/py-faster-rcnn \
-v /data/train_data:/storage/data \
-v /$(pwd):/opt/ros-ort \
-it \
--workdir="//opt/ros-ort" \
-e ROS_MASTER_URI=http://roscore_kinetic:11311/ \
--env="DISPLAY" \
--env="QT_X11_NO_MITSHM=1" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
--name ros_frcnn_training \
--entrypoint="/train_entrypoint.sh" \
meppe78/ros-kinetic-frcnn-training \
/usr/bin/python src/frcnn/scripts/run_training.py --set DATA_DIR /storage/data

