#!/bin/bash
/usr/sbin/sshd -D &

set -e

# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash"
export PYTHONPATH="$PYTHONPATH:/opt/ros/$ROS_DISTRO/lib/python2.7/dist-packages"
export ROS_IP=`hostname -I`
export ROS_MASTER_URI=http://roscore_kinetic:11311
exec "$@"
