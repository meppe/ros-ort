#!/bin/bash

gnome-terminal -e ./run_roscore.sh

sleep 1

gnome-terminal -e ./run_video_stream.sh

sleep 1

gnome-terminal -e "./run_detect.sh --gpu"

sleep 1

gnome-terminal -e ./run_track.sh

sleep 1

gnome-terminal -e ./run_video_tracking_view.sh

sleep 1

gnome-terminal -e ./run_txt_interface.sh
