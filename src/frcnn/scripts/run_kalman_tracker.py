#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import argparse

### KALMAN TRACKER NOT YET WORKING PROPERLY!!!!

from frcnn.kalman_tracker import KalmanTracker

if __name__ == '__main__':
    kalman_tracker = KalmanTracker()

