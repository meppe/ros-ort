#!/bin/bash

gnome-terminal -e ./run_roscore.sh

sleep 4

gnome-terminal -e ./run_video_stream.sh

sleep 4

gnome-terminal -e "./run_detect.sh --gpu"

sleep 4

gnome-terminal -e ./run_track.sh

sleep 4

gnome-terminal -e ./run_video_tracking_view.sh

sleep 4

gnome-terminal -e ./run_txt_interface.sh
