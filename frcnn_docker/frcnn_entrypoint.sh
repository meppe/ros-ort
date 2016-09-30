#!/bin/bash
/usr/sbin/sshd

set -e

# setup ros environment
# source "/opt/ros/$ROS_DISTRO/setup.bash"
export ROS_IP=`hostname -I`
exec "$@"
