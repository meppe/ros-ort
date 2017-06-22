#!/bin/bash
# This script is to show detection in some sample images. No videos, no streaming, just to verify training.
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' ros_frcnn_demo`

if [ "$1" = "--cpu" ]; then
    docker-compose run detect_demo_cpu
    echo "Running demo in cpu mode"
else
    echo "Running demo in gpu mode"
    nvidia-docker-compose run detect_demo_gpu
fi
xhost -local:`docker inspect --format='{{ .Config.Hostname }}' ros_frcnn_demo`