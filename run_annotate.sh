#!/bin/bash
# Only allow the container with the ID (hostname) of ros_video_view to control my X-Server
image_name="annotate_frames"
docker rm $image_name
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' $image_name`
docker run \
--rm \
-v /$(pwd):/opt/ros-ort \
-v /storage/data:/storage/data \
-v /$(pwd)/src/frcnn/src/py-faster-rcnn:/opt/ros-ort/src/frcnn/src/py-faster-rcnn \
-v /$(pwd)/src/frcnn:/opt/ros-ort/src/frcnn \
-it \
--workdir="//opt/ros-ort" \
--name $image_name \
--env="DISPLAY" \
--env="QT_X11_NO_MITSHM=1" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
meppe78/ros-kinetic-image-view \
bash -c 'python src/frcnn/scripts/run_annotate.py'

#docker-compose run annotate

xhost -local:`docker inspect --format='{{ .Config.Hostname }}' $image_name`

