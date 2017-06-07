#!/bin/bash

set -e

# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash"
source "/opt/ros-ort/devel/setup.bash"
export PYTHONPATH="$PYTHONPATH:/opt/ros/$ROS_DISTRO/lib/python2.7/dist-packages"
# export PYTHONPATH="$PYTHONPATH:/opt/ros-ort/src/frcnn/src"
export ROS_MASTER_URI=http://roscore_kinetic:11311
export ROS_IP=`hostname -I`
exec "$@"
