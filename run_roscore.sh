#!/bin/bash



docker rm roscore_kinetic
docker run \
-it \
--rm \
--name roscore_kinetic \
--network bridge \
meppe78/ros-core-kinetic \
roscore




