#!/bin/bash
docker rm ros_frcnn_training
echo "Note: Training must be run with nvidia-docker and GPU-support because some layers in caffe do not support cpu mode. Running nvidia-docker with GPU"
docker pull meppe78/ros-kinetic-frcnn-training
nvidia-docker-compose run training



