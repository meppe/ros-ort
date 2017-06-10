#!/bin/bash
docker rm ros_frcnn_detect
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' ros_frcnn_detect`
### If using ZF models, run with a very low conf threshold
if [ "$1" = "--gpu" ]; then
	echo "Running nvidia-docker with GPU"
	nvidia-docker run \
	--rm \
	-v /$(pwd)/src/frcnn:/opt/ros-ort/src/frcnn \
	-v /$(pwd)/src/ort_msgs:/opt/ros-ort/src/ort_msgs \
	-v /$(pwd)/src/frcnn/src/py-faster-rcnn:/opt/ros-ort/src/frcnn/src/py-faster-rcnn \
	-v /$(pwd)/output:/opt/ros-ort/output \
	-it \
	--workdir="//opt/ros-ort" \
	--link roscore_kinetic \
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
				&& python src/frcnn/scripts/run_detect.py --threshold 0.005 --model nico_vgg16"
else
	echo "Running docker with CPU"
	docker run \
	--rm \
	-v /$(pwd)/src/frcnn:/opt/ros-ort/src/frcnn \
	-v /$(pwd)/src/ort_msgs:/opt/ros-ort/src/ort_msgs \
	-v /$(pwd)/output:/opt/ros-ort/output \
	-it \
	--workdir="//opt/ros-ort" \
	--link roscore_kinetic \
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
				&& python src/frcnn/scripts/run_detect.py --cpu "
fi

xhost -local:`docker inspect --format='{{ .Config.Hostname }}' ros_frcnn_detect`
