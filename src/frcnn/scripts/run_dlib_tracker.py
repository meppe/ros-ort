#!/usr/bin/env python


import argparse

from frcnn.dlib_tracker import DlibTracker
import argparse


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tracking module for ros-ort')
    parser.add_argument('--cum_threshold', dest='cum_threshold', help='The cumulative threshold over time to keep a tracker',
                        default=0.3, type=float)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    tracker = DlibTracker(args)

