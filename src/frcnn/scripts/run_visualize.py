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

from frcnn.visualizer import Visualizer

# NETS = {'vgg16': ('VGG16',
#                   'VGG16_faster_rcnn_final.caffemodel'),
#         'zf': ('ZF',
#                   'ZF_faster_rcnn_final.caffemodel')}

# CLASSES = ('__background__',
#                    'aeroplane', 'bicycle', 'bird', 'boat',
#                    'bottle', 'bus', 'car', 'cat', 'chair',
#                    'cow', 'diningtable', 'dog', 'horse',
#                    'motorbike', 'person', 'pottedplant',
#                    'sheep', 'sofa', 'train', 'tvmonitor')

# def parse_args():
#     """Parse input arguments."""
#     parser = argparse.ArgumentParser(description='Faster R-CNN demo')
#     parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
#                         default=0, type=int)
#     parser.add_argument('--cpu', dest='cpu_mode',
#                         help='Use CPU mode (overrides --gpu)',
#                         action='store_true')
#     parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
#                         choices=NETS.keys(), default='vgg16')
#
#     args = parser.parse_args()
#
#     return args

if __name__ == '__main__':
    # args = parse_args()
    # detector = Detector(args, NETS, CLASSES)
    visualizer = Visualizer()

