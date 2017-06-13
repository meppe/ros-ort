#!/bin/bash
docker rm ros_video_stream
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' ros_video_stream`
echo "The first and only argument to this script is the relative filepath of the video inside the container"
if [ $# -eq 0 ]; then
    filepath="video_data/security_cam_transporter.mp4"
else
    filepath=$1
fi
working_dir="/opt/ros-ort"
docker run \
--rm \
-v /$(pwd)/src/video_stream_opencv:/opt/ros-ort/src/video_stream_opencv \
-v /$(pwd)/src/ros-ort/launch:/opt/ros-ort/src/ros-ort/launch \
-v /$(pwd)/video_data:/opt/ros-ort/video_data \
-it \
--workdir=$working_dir \
--link roscore_kinetic \
-e ROS_MASTER_URI=http://roscore_kinetic:11311/ \
--env="DISPLAY" \
--env="QT_X11_NO_MITSHM=1" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
--device="/dev/video0" \
--name ros_video_stream \
meppe78/ros-kinetic-video-stream \
bash -c "source '/opt/ros/kinetic/setup.bash' && source '/opt/ros-ort/devel/setup.bash' && \
			roslaunch src/ros-ort/launch/camera.launch visualize:=true camera_name:=frcnn_input"
#bash -c "source '/opt/ros/kinetic/setup.bash' && source '/opt/ros-ort/devel/setup.bash' && \
#			roslaunch src/video_stream_opencv/launch/camera.launch \
#			visualize:=false \
#			video_stream_provider:="$working_dir"/"$filepath" \
#			camera_name:=frcnn_input \
#			fps:=10"
# bash -c "source '/opt/ros/kinetic/setup.bash' && source '/opt/ros-ort/devel/setup.bash' && \
# 			roslaunch src/video_stream_opencv/launch/camera.launch \
# 			visualize:=false \
# 			video_stream_provider:=/opt/ros-ort/video_data/Goodfellas_Long_Take_Restaurant_640.mp4 \
# 			camera_name:=frcnn_input \
# 			fps:=2"	
# bash -c "source '/opt/ros/kinetic/setup.bash' && source '/opt/ros-ort/devel/setup.bash' && \
# 			roslaunch src/video_stream_opencv/launch/camera.launch \
# 			visualize:=false \
# 			video_stream_provider:=/opt/ros-ort/video_data/Gravity_Long_Take_Shorter_640.mp4 \
# 			camera_name:=frcnn_input \
# 			fps:=2"			
# bash -c "source '/opt/ros/kinetic/setup.bash' && source '/opt/ros-ort/devel/setup.bash' && \
# 			roslaunch src/video_stream_opencv/launch/camera.launch \
# 			visualize:=false \
# 			video_stream_provider:=/opt/ros-ort/video_data/security_cam_transporter.mp4 \
# 			camera_name:=frcnn_input \
# 			fps:=2"	
# bash -c "source '/opt/ros/kinetic/setup.bash' && source '/opt/ros-ort/devel/setup.bash' && \
# 			roslaunch src/video_stream_opencv/launch/camera.launch \
# 			visualize:=false \
# 			video_stream_provider:=/opt/ros-ort/video_data/pedestrians_street.mp4 \
# 			camera_name:=frcnn_input \
# 			fps:=1"				
# bash -c "source '/opt/ros/kinetic/setup.bash' && source '/opt/ros-ort/devel/setup.bash' && \
# 		rosbag play -l video_data/LSD_room.bag -r 0.02 -l /image_raw:=/frcnn_input/image_raw"
	
xhost -local:`docker inspect --format='{{ .Config.Hostname }}' ros_video_stream`