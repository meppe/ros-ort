#!/bin/bash
img_name="ros_frcnn_demo"
docker rm "$img_name"
if [ "$1" = "--cpu" ]; then
    command="docker"
    args="--cpu"
    echo "Running demo in cpu mode"
else
    command="nvidia-docker"
    echo "Running demo in gpu mode"
    args=""
fi
$command run \
--rm \
-v /$(pwd)/src/frcnn:/opt/ros-ort/src/frcnn \
-v /$(pwd)/src/ort_msgs:/opt/ros-ort/src/ort_msgs \
-v /$(pwd)/output:/opt/ros-ort/output \
-v /$(pwd)/src/frcnn/src/py-faster-rcnn:/opt/ros-ort/src/frcnn/src/py-faster-rcnn \
-v /storage/data:/storage/data \
-v /$(pwd):/opt/ros-ort \
-it \
--workdir="//opt/ros-ort" \
-e ROS_MASTER_URI=http://roscore_kinetic:11311/ \
--env="DISPLAY" \
--env="QT_X11_NO_MITSHM=1" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
--name="$img_name" \
--entrypoint="/dummy_entrypoint.sh" \
meppe78/ros-kinetic-frcnn-training \
bash -c "cp -rn /py-faster-rcnn /opt/ros-ort/src/frcnn/src/ \
        && python src/frcnn/scripts/run_demo.py $args --set DATA_DIR /storage/data"
