#!/usr/bin/env python


import argparse
import _init_paths
from frcnn.dlib_tracker import DlibTracker
import argparse


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tracking module for ros-ort')
    parser.add_argument('--cum_threshold', dest='cum_threshold', help='The cumulative threshold over time to keep a tracker',
                        default=1.5, type=float)
    parser.add_argument('--class_threshold', dest='class_threshold',
                        help='The threshold for a single class over time',
                        default=1, type=float)
    parser.add_argument('--max_trackers', dest='max_trackers',
                        help='The maximum number of objects to track',
                        default=5, type=int)
    parser.add_argument('--mask_objects', dest='mask_objects', help='Whether or not to display objects only and to remove background',
                        default=False, type=bool)
    parser.add_argument('--write_to_img_file', dest='write_to_img_file',
                        help='Whether or not to write each frame image with BBs to the file specified',
                        default='/tmp/img.jpg', type=str)
    parser.add_argument('--write_to_detections_file', dest='write_to_detections_file',
                        help='Whether or not to write each frame\'s bounding boxes to the file specified',
                        default='/tmp/detections.txt', type=str)


    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    tracker = DlibTracker(args)

