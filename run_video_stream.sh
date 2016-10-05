#!/bin/bash
docker rm ros_video_stream
docker run \
--rm \
-v /$(pwd):/opt/ros-ort \
-it \
--workdir="//opt/ros-ort" \
--link roscore_kinetic \
-e ROS_MASTER_URI=http://roscore_kinetic:11311/ \
--name ros_video_stream \
meppe78/ros-core-kinetic \
bash -c 'rosbag play -l video_data/LSD_room.bag -r 0.1'