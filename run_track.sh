#!/bin/bash
docker rm ros_frcnn_tracking
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' ros_frcnn_tracking`
docker run \
--rm \
-v /$(pwd)/src/frcnn:/opt/ros-ort/src/frcnn \
-v /$(pwd)/src/ort_msgs:/opt/ros-ort/src/ort_msgs \
-v /$(pwd)/output:/opt/ros-ort/output \
-v /tmp:/tmp \
-it \
--workdir="//opt/ros-ort" \
--link roscore_kinetic \
-e ROS_MASTER_URI=http://roscore_kinetic:11311/ \
--env="DISPLAY" \
--env="QT_X11_NO_MITSHM=1" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
--name ros_frcnn_tracking \
--entrypoint="/frcnn_entrypoint.sh" \
meppe78/ros-kinetic-frcnn-tracker \
python src/frcnn/scripts/run_dlib_tracker.py --cum_threshold 1.5 --class_threshold 1 --max_trackers 5
xhost -local:`docker inspect --format='{{ .Config.Hostname }}' ros_frcnn_tracking`
