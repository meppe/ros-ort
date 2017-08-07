#!/bin/bash
docker rm ros_tracking_record
echo "In the container, run 'rosbag record -j frcnn/bb_img_tracking' to start recording."
docker run \
--rm \
-v /$(pwd):/opt/ros-ort \
-it \
--workdir="//opt/ros-ort" \
--link roscore_kinetic \
-e ROS_MASTER_URI=http://roscore_kinetic:11311/ \
--name ros_tracking_record \
meppe78/ros-core-kinetic \
bash
