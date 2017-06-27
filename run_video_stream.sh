#!/bin/bash
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' ros_video_stream`
echo "The first and only argument to this script is the relative filepath of the video inside the container. If no argument is given, the script will publish from /dev/video0"
working_dir="/opt/ros-ort"
if [ $# -eq 0 ]; then
    echo "Running stream from /dev/video0"
    export ROS_ORT_VIDEO=""
else
    export ROS_ORT_VIDEO="$working_dir"/"$1"
    echo "Running stream from $ROS_ORT_VIDEO"
fi
docker-compose run video_stream

xhost -local:`docker inspect --format='{{ .Config.Hostname }}' ros_video_stream`