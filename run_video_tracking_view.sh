#!/bin/bash
# Only allow the container with the ID (hostname) of ros_video_view to control my X-Server
docker rm ros_video_tracking_view
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' ros_video_tracking_view` 
docker run \
--rm \
-v /$(pwd):/opt/semslam \
-it \
--workdir="//opt/ros-ort" \
--link roscore_kinetic \
--name ros_video_tracking_view \
-e ROS_MASTER_URI=http://roscore_kinetic:11311/ \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
meppe78/ros-kinetic-image-view \
bash -c 'rosrun image_view image_view image:=/frcnn/bb_img_tracking'
xhost -local:`docker inspect --format='{{ .Config.Hostname }}' ros_video_tracking_view`