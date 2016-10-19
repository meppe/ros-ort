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

from frcnn.detector import Detector

NETS_PASCAL = {'vgg16': ('VGG16', 'faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel'),
               'zf': ('ZF', 'faster_rcnn_models/ZF_faster_rcnn_final.caffemodel')}

NETS_COCO = {'vgg16': ('VGG16', 'imagenet_models/VGG16.v2.caffemodel'),
             'zf': ('ZF', 'imagenet_models/ZF.v2.caffemodel')}

CLASSES_PASCAL = ('__background__',
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')

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

MODELS = {'coco': ('coco', 'faster_rcnn_end2end', 'test.prototxt', CLASSES_COCO, NETS_COCO),
          'pascal': ('pascal_voc', 'faster_rcnn_alt_opt', 'faster_rcnn_test.pt', CLASSES_PASCAL, NETS_PASCAL)}


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)

    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')

    # parser.add_argument('--net', dest='net', help='Network to use [vgg16]',
    #                     choices=NETS.keys(), default='vgg16')

    poss_models = []
    for m in MODELS:
        for n in MODELS[m][4].keys():
            poss_models.append(m+"--"+n)

    parser.add_argument('--model', dest='model', help='Model to use [pascal]',
                        choices=poss_models, default='pascal--vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    detector = Detector(args, MODELS)
