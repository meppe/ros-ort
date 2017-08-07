#!/bin/bash
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' ros_video_stream`
docker pull meppe78/ros-kinetic-video-stream
docker rm ros_video_stream
echo "The first and only argument to this script is the relative filepath of the video inside the container. \
        If no argument is given, the script will publish from /dev/video0"
working_dir="/opt/ros-ort"
if [ $# -eq 0 ]; then
    filepath=""
    echo "Running stream from /dev/video0"
else
    filepath=$working_dir"/"$1
    echo "Running stream from $filepath"
fi

docker run \
--rm \
-v /$(pwd)/src/video_stream_opencv:/opt/ros-ort/src/video_stream_opencv \
-v /$(pwd)/src/ros-ort/launch:/opt/ros-ort/src/ros-ort/launch \
-v /$(pwd)/video_data:/opt/ros-ort/video_data \
-it \
--workdir=$working_dir \
-e ROS_MASTER_URI=http://134.100.10.141:11311/ \
--env="DISPLAY" \
--env="QT_X11_NO_MITSHM=1" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
--device="/dev/video0" \
--name ros_video_stream \
meppe78/ros-kinetic-video-stream \
           roslaunch src/video_stream_opencv/launch/camera.launch \
            visualize:=false \
            video_stream_provider:="$filepath" \
            camera_name:=frcnn_input \
            fps:=1

xhost -local:`docker inspect --format='{{ .Config.Hostname }}' ros_video_stream`
