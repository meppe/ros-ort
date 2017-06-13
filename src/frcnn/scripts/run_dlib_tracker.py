#!/usr/bin/env python


import argparse
import _init_paths
from frcnn.dlib_tracker import DlibTracker
import argparse


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tracking module for ros-ort')
    parser.add_argument('--cum_threshold', dest='cum_threshold', help='The cumulative threshold over time to keep a tracker',
                        default=0.3, type=float)
    parser.add_argument('--class_threshold', dest='class_threshold',
                        help='The threshold for a single class over time',
                        default=0.1, type=float)
    parser.add_argument('--mask_objects', dest='mask_objects', help='Whether or not to display objects only and to remove background',
                        default=False, type=bool)
    parser.add_argument('--write_to_file', dest='write_to_file',
                        help='Whether or not to write each frame to the file specified',
                        default='', type=str)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    tracker = DlibTracker(args)

