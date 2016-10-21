#!/bin/bash
docker rm ros_video_stream
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' ros_video_stream` 
docker run \
--rm \
-v /$(pwd)/src/video_stream_opencv:/opt/ros-ort/src/video_stream_opencv \
-v /$(pwd)/video_data:/opt/ros-ort/video_data \
-it \
--workdir="//opt/ros-ort" \
--link roscore_kinetic \
-e ROS_MASTER_URI=http://roscore_kinetic:11311/ \
--env="DISPLAY" \
--env="QT_X11_NO_MITSHM=1" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
--name ros_video_stream \
meppe78/ros-kinetic-video-stream \
bash -c "source '/opt/ros/kinetic/setup.bash' && source '/opt/ros-ort/devel/setup.bash' && \
			roslaunch src/video_stream_opencv/launch/camera.launch \
			visualize:=false \
			video_stream_provider:=/opt/ros-ort/video_data/security_cam_transporter.mp4 \
			camera_name:=frcnn_input \
			fps:=5"
# bash -c 'rosbag record -j frcnn/bb_img_tracking frcnn/bb frcnn/bb_img'
xhost -local:`docker inspect --format='{{ .Config.Hostname }}' ros_video_stream`