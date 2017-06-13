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
# import sys
import _init_paths

# sys.path.append("/opt/ros-ort/devel/lib/python2.7/dist-packages")
# sys.path.append("/opt/ros/kinetic/lib/python2.7/dist-packages")
# sys.path.append("/opt/ros-ort/install/lib/python2.7/dist-packages")
# sys.path.append("/opt/ros-ort/src/frcnn/src")
# sys.path.append("/opt/ros/kinetic/bin")

# print sys.path
from frcnn.detector import Detector
from lib.datasets.nico import Nico
from lib.datasets.pascal_voc import pascal_voc


CLASSES_NICO = Nico.CLASSES

CLASSES_PASCAL = pascal_voc.CLASSES

CLASSES_COCO = ('__background__',
                   'aeroplane0', 'bicycle0', 'bird0', 'boat0',
                   'bottle0', 'bus0', 'car0', 'cat0', 'chair0',
                   'cow0', 'diningtable0', 'dog0', 'horse0',
                   'motorbike0', 'person0', 'pottedplant0',
                   'sheep0', 'sofa0', 'train0', 'tvmonitor0',
                    'aeroplane1', 'bicycle1', 'bird1', 'boat1',
                    'bottle1', 'bus1', 'car1', 'cat1', 'chair1',
                    'cow1', 'diningtable1', 'dog1', 'horse1',
                    'motorbike1', 'person1', 'pottedplant1',
                    'sheep1', 'sofa1', 'train1', 'tvmonitor1',
                    'aeroplane2', 'bicycle2', 'bird2', 'boat2',
                    'bottle2', 'bus2', 'car2', 'cat2', 'chair2',
                    'cow2', 'diningtable2', 'dog2', 'horse2',
                    'motorbike2', 'person2', 'pottedplant2',
                    'sheep2', 'sofa2', 'train2', 'tvmonitor2',
                    'aeroplane3', 'bicycle3', 'bird3', 'boat3',
                    'bottle3', 'bus3', 'car3', 'cat3', 'chair3',
                    'cow3', 'diningtable3', 'dog3', 'horse3',
                    'motorbike3', 'person3', 'pottedplant3',
                    'sheep3', 'sofa3', 'train3', 'tvmonitor3')

MODELS = {
            # 'coco': ('coco',
            #        'faster_rcnn_end2end',
            #        'test.prototxt',
            #        CLASSES_COCO, NETS_COCO),
          'pascal_vgg16': (
                     'src/frcnn/src/data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel',
                     'src/frcnn/src/models/pascal_voc/VGG16/faster_rcnn_end2end/test.prototxt',
                     CLASSES_PASCAL),
          'pascal_zf': (
                     'src/frcnn/src/data/faster_rcnn_models/ZF_faster_rcnn_final.caffemodel',
                     'src/frcnn/src/models/pascal_voc/ZF/faster_rcnn_end2end/test.prototxt',
                     CLASSES_PASCAL),
          'nico_zf': (
                   'src/frcnn/src/output/faster_rcnn_end2end/nico_2017_trainval/zf_faster_rcnn_iter_30000.caffemodel',
                   'src/frcnn/src/models/nico/ZF/faster_rcnn_end2end/test.prototxt',
                   CLASSES_NICO),
          'nico_vgg16': (
                   'src/frcnn/src/output/faster_rcnn_end2end/nico_2017_trainval/vgg16_faster_rcnn_iter_70000.caffemodel',
                   'src/frcnn/src/models/nico/VGG16/faster_rcnn_end2end/test.prototxt',
                   CLASSES_NICO),
          }

# BASE_DIR = "/opt/ros-ort"

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)

    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')

    parser.add_argument('--threshold', dest='conf_threshold',
                        help='The confidence threshold to detect an object', default=0.4)
    

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

    detector = Detector(classes, prototxt_file, caffemodel_file, args)
