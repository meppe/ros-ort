#!/bin/bash
# docker rm ros_frcnn
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' ros_frcnn` 
nvidia-docker run \
--rm \
-v /$(pwd)/src/frcnn:/opt/ros-ort/src/frcnn \
-v /$(pwd)/src/ort_msgs:/opt/ros-ort/src/ort_msgs \
-v /$(pwd)/output:/opt/ros-ort/output \
-it \
--workdir="//opt/ros-ort" \
--link roscore_kinetic \
-e ROS_MASTER_URI=http://roscore_kinetic:11311/ \
--env="DISPLAY" \
--env="QT_X11_NO_MITSHM=1" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
--name ros_frcnn \
--entrypoint="/frcnn_entrypoint.sh" \
meppe78/ros-kinetic-frcnn \
bash  
xhost -local:`docker inspect --format='{{ .Config.Hostname }}' ros_frcnn`