#!/bin/bash
if [ "$1" = "--gpu" ]; then
    echo "Running detection on GPU"
    nvidia-docker-compose run detection_gpu
else
    echo "Running detection on CPU"
    docker-compose run detection_cpu
fi

