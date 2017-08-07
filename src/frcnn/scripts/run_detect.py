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

# Disable X-output for matplotlib
import matplotlib
matplotlib.use('pdf')

import argparse

import _init_paths

from frcnn.detector import Detector
from lib.datasets.nico import Nico
from lib.datasets.pascal_voc import pascal_voc


CLASSES_NICO = Nico.CLASSES
CLASS_PROPERTIES_NICO = Nico.CLASS_PROPERTIES

CLASSES_PASCAL = pascal_voc.CLASSES

MODELS = {
          'pascal_vgg16': (
                     'caffemodels/faster_rcnn_models/VGG16_faster_rcnn_nico.caffemodel',
                     'src/frcnn/src/models/pascal_voc/VGG16/faster_rcnn_end2end/test.prototxt',
                     CLASSES_PASCAL),
          'pascal_zf': (
                     'caffemodels/faster_rcnn_models/VGG16_faster_rcnn_nico.caffemodel',
                     'src/frcnn/src/models/pascal_voc/ZF/faster_rcnn_end2end/test.prototxt',
                     CLASSES_PASCAL),
          'nico_zf': (
                   'caffemodels/nico_frcnn_models/ZF_faster_rcnn_nico.caffemodel',
                   'src/frcnn/src/models/nico/ZF/faster_rcnn_end2end/test.prototxt',
                   CLASSES_NICO, CLASS_PROPERTIES_NICO),
          'nico_vgg16': (
                   'caffemodels/nico_frcnn_models/VGG16_faster_rcnn_nico.caffemodel',
                   'src/frcnn/src/models/nico/VGG16/faster_rcnn_end2end/test.prototxt',
                   CLASSES_NICO, CLASS_PROPERTIES_NICO),
          }

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)

    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')

    parser.add_argument('--threshold', dest='conf_threshold',
                        help='The confidence threshold to detect an object', default=0.04, type=float)
    
    poss_models = []
    for m in MODELS.keys():
        poss_models.append(m)

    parser.add_argument('--model', dest='model', help='Model to use [pascal]',
                        choices=poss_models, default='pascal_vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    model = args.model
    caffemodel_file = MODELS[model][0]
    prototxt_file = MODELS[model][1]
    classes = MODELS[model][2]
    if len(MODELS[model]) == 4:
        class_properties = MODELS[model][3]
    detector = Detector(classes, prototxt_file, caffemodel_file, args, class_properties=class_properties)
