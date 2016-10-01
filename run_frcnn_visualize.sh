#!/bin/bash
docker rm ros_frcnn_visualize
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' ros_frcnn_visualize` 
nvidia-docker run \
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
--name ros_frcnn_visualize \
--entrypoint="/frcnn_entrypoint.sh" \
meppe78/ros-kinetic-frcnn \
bash
# bash -c "source '/opt/ros/kinetic/setup.bash' && source '/opt/ros-ort/devel/setup.bash' && python src/frcnn/scripts/run_visualize.py --cpu"
xhost -local:`docker inspect --format='{{ .Config.Hostname }}' ros_frcnn_visualize`