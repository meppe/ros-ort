#!/bin/bash

gnome-terminal -e ./run_roscore.sh

gnome-terminal -e ./run_video_stream.sh

gnome-terminal -e "./run_detect.sh --gpu"

gnome-terminal -e ./run_track.sh

gnome-terminal -e ./run_video_tracking_view.sh

gnome-terminal -e ./run_txt_interface.sh
