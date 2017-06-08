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

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from lib.datasets.pascal_voc import pascal_voc
from lib.datasets.coco import coco
from lib.datasets.nico import Nico

# CLASSES = ('__background__',
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat', 'chair',
#            'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

base_dir = "/opt/ros-ort/src/frcnn/src/"
# CLASSES = ()

def parse_args():
    """Parse input arguments."""
    nets = ["ZF, VGG16"]
    datasets = ["nico", "pascal_voc"]
    methods = ["faster_rcnn_end2end", "faster_rcnn_alt_opt"]
    train_imdbs = ["nico_2017_trainval", "voc_2007_trainval"]
    test_dirs = ["nico2017/nico2017", "VOCdevkit2007/VOC2007"]

    net = "ZF"
    dataset = "pascal_voc"
    method = "faster_rcnn_end2end"
    train_imdb = "voc_2007_trainval"
    test_dir = "VOCdevkit2007/VOC2007"

    model = "700"
    caffemodel = net.lower() + "_faster_rcnn_iter_" + str(model) + ".caffemodel"

    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='net', help='Network to use',
                        choices=nets, default=net)
    parser.add_argument('--dataset', dest='dataset',
                        help='dataset to test with',
                        default=dataset, type=str, choices=datasets)
    parser.add_argument('--method', dest='method',
                        help='the method with which was trained',
                        default=method, type=str, choices=methods)
    parser.add_argument('--imdb', dest='train_imdb',
                        help='dataset to train on',
                        default=train_imdb, type=str, choices=train_imdbs)
    parser.add_argument('--caffemodel', dest='caffemodel',
                        help='caffemodel file',
                        default=caffemodel, type=str)
    parser.add_argument('--test_dirs', dest='test_dir',
                        help='directory that contains testing data',
                        default=test_dir, type=str, choices=test_dirs)

    args = parser.parse_args()

    global CLASSES
    if args.dataset == "nico":
        CLASSES = Nico.CLASSES
    elif args.dataset == "pascal_voc":
        CLASSES = pascal_voc.CLASSES

    return args


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(net, im_file):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    # CONF_THRESH = 0.8
    # NMS_THRESH = 0.3
    CONF_THRESH = 0.01
    NMS_THRESH = 0.01
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)



if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()
    # prototxt = os.path.join(cfg.MODELS_DIR, NETS["vgg16"][0],
    #                         'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    prototxt = base_dir + 'models/' + args.dataset + '/' + args.net + '/faster_rcnn_end2end/test.prototxt'
    if not os.path.isfile(prototxt):
        raise IOError('{:s} not found.'.format(prototxt))

    # caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
    #                           NETS["vgg16"][1])
    caffemodel = os.path.join(base_dir, "output", args.method, args.train_imdb, args.caffemodel)

    if not os.path.isfile(caffemodel):
        raise IOError('{:s} not found.'.format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    im_names = ['000001.jpg', '000002.jpg', '000003.jpg', '000004.jpg']
    for im_name in im_names:
        im_file = os.path.join("/storage", "data", args.test_dir, "JPEGImages", im_name)
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for {}'.format(im_file)
        demo(net, im_file)

    plt.show()
