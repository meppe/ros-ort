#!/bin/bash
# Only allow the container with the ID (hostname) of ros_video_view to control my X-Server
docker rm ros_video_view

xhost +local:`docker inspect --format='{{ .Config.Hostname }}' ros_video_view`

docker run \
--rm \
-v /$(pwd):/opt/ros-ort \
-v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
-it \
--workdir="//opt/ros-ort" \
--link roscore_kinetic:latest \
--name ros_video_view \
-e ROS_MASTER_URI=http://roscore_kinetic:11311/ \
-e DISPLAY \
-e "QT_X11_NO_MITSHM=1" \
meppe78/ros-kinetic-image-view \
rosrun image_view image_view image:=/frcnn_input/image_raw

xhost -local:`docker inspect --format='{{ .Config.Hostname }}' ros_video_view`
