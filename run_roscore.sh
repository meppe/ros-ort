#!/bin/bash



docker rm roscore_kinetic
docker run \
-it \
--rm \
--name roscore_kinetic \
-p 11311:11311 \
meppe78/ros-core-kinetic \
roscore




