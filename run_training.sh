#!/bin/bash
docker rm ros_frcnn_detect
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' ros_frcnn_detect` 
if [ "$1" = "--cpu" ]; then
	echo echo "Running docker with CPU"
	nvidia-docker run \
	--rm \
	-v /$(pwd)/src/frcnn:/opt/ros-ort/src/frcnn \
	-v /$(pwd)/src/ort_msgs:/opt/ros-ort/src/ort_msgs \
	-v /$(pwd)/src/frcnn/src/py-faster-rcnn:/opt/ros-ort/src/frcnn/src/py-faster-rcnn \
	-v /$(pwd)/output:/opt/ros-ort/output \
	-v /storage/data:/storage/data \
	-it \
	--workdir="//opt/ros-ort" \
	-e ROS_MASTER_URI=http://roscore_kinetic:11311/ \
	--env="DISPLAY" \
	--env="QT_X11_NO_MITSHM=1" \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--name ros_frcnn_detect \
	--entrypoint="/frcnn_entrypoint.sh" \
	meppe78/ros-kinetic-frcnn \
	bash -c "cp -rn /py-faster-rcnn /opt/ros-ort/src/frcnn/src/ \
				&& source '/opt/ros/kinetic/setup.bash' \
				&& source '/opt/ros-ort/devel/setup.bash' \
				&& python src/frcnn/scripts/run_training.py --cpu --set DATA_DIR /storage/data"
else
	echo "Running nvidia-docker with GPU"
	nvidia-docker run \
	--rm \
	-v /$(pwd)/src/frcnn:/opt/ros-ort/src/frcnn \
	-v /$(pwd)/src/ort_msgs:/opt/ros-ort/src/ort_msgs \
	-v /$(pwd)/output:/opt/ros-ort/output \
	-v /storage/data:/storage/data \
	-it \
	--workdir="//opt/ros-ort" \
	-e ROS_MASTER_URI=http://roscore_kinetic:11311/ \
	--env="DISPLAY" \
	--env="QT_X11_NO_MITSHM=1" \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--name ros_frcnn_detect \
	--entrypoint="/frcnn_entrypoint.sh" \
	meppe78/ros-kinetic-frcnn \
	bash -c "cp -rn /py-faster-rcnn /opt/ros-ort/src/frcnn/src/ \
				&& source '/opt/ros/kinetic/setup.bash' \
				&& source '/opt/ros-ort/devel/setup.bash' \
				&& python src/frcnn/scripts/run_training.py --set DATA_DIR /storage/data"
fi

xhost -local:`docker inspect --format='{{ .Config.Hostname }}' ros_frcnn_detect`
