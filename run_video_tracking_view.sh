#!/bin/bash
# Only allow the container with the ID (hostname) of ros_video_view to control my X-Server
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' ros_video_tracking_view`

docker-compose run tracking_view

xhost -local:`docker inspect --format='{{ .Config.Hostname }}' ros_video_tracking_view`