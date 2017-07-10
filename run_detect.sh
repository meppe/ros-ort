#!/bin/bash
docker pull meppe78/ros-kinetic-frcnn
docker rm ros_frcnn_detect
### If using ZF models, run with a very low conf threshold
if [ "$1" = "--gpu" ]; then
    echo "Running detection on GPU"
    command="docker"
    py_args=""
    nv_driver=375.26
    args=" --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm -v /var/lib/nvidia-docker/volumes/nvidia_driver/${nv_driver}:/usr/local/nvidia:ro"
else
    echo "Running detection on CPU"
    command="docker"
    py_args="--cpu"
    args=""
fi
	$command run \
	$args \
	--rm \
	-v /$(pwd)/src/frcnn:/opt/ros-ort/src/frcnn \
	-v /$(pwd)/src/ort_msgs:/opt/ros-ort/src/ort_msgs \
	-v /$(pwd)/src/frcnn/src/py-faster-rcnn:/opt/ros-ort/src/frcnn/src/py-faster-rcnn \
	-v /$(pwd)/output:/opt/ros-ort/output \
	-v /$(pwd)/caffemodels:/opt/ros-ort/caffemodels \
	-it \
	--workdir="//opt/ros-ort" \
	--link roscore_kinetic \
	-e ROS_MASTER_URI=http://roscore_kinetic:11311/ \
	--name ros_frcnn_detect \
	--entrypoint="/frcnn_entrypoint.sh" \
	meppe78/ros-kinetic-frcnn \
	python src/frcnn/scripts/run_detect.py --threshold 0.1 --model nico_zf $py_args


