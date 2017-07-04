#!/bin/bash
#
#
#docker-machine create \
#  -d virtualbox \
#  kv
#
#docker $(docker-machine config kv) run -d \
#  -p 8500:8500 -h consul \
#  progrium/consul \
#  -server -bootstrap
#
#docker-machine create \
#  -d virtualbox \
#  --swarm \
#  --swarm-master \
#  --swarm-discovery="consul://127.0.0.1:8500" \
#  --engine-opt="cluster-store=consul://127.0.0.1:8500" \
#  --engine-opt="cluster-advertise=eth1:2376" \
#  swarm-master
#
#
#docker network create -d overlay multihost --attachable
#

docker rm roscore_kinetic
docker run \
-it \
--rm \
--name roscore_kinetic \
meppe78/ros-core-kinetic \
roscore




