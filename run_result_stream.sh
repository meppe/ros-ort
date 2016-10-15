#!/bin/bash
# docker rm ros_result_stream
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' ros_result_stream` 
docker run \
--rm \
-v /$(pwd)/src/video_stream_opencv:/opt/ros-ort/src/video_stream_opencv \
-v /$(pwd)/video_data:/opt/ros-ort/video_data \
-v /$(pwd)/results:/opt/ros-ort/results \
-it \
--workdir="//opt/ros-ort" \
--link roscore_kinetic \
-e ROS_MASTER_URI=http://roscore_kinetic:11311/ \
--env="DISPLAY" \
--env="QT_X11_NO_MITSHM=1" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
--name ros_result_stream \
meppe78/ros-kinetic-video-stream \
bash -c "source '/opt/ros/kinetic/setup.bash' && source '/opt/ros-ort/devel/setup.bash' && \
			rosbag play -r 30 -l results/Transporter_tracking_frcnn-detect-clustered_fps=1.bag"
xhost -local:`docker inspect --format='{{ .Config.Hostname }}' ros_result_stream`