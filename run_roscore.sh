#!/bin/bash
docker rm roscore_kinetic
docker run \
-it \
--rm \
--name roscore_kinetic \
meppe78/ros-core-kinetic \
roscore
