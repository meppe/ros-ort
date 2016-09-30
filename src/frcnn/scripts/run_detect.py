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
print ("starting detection")

import rospkg
from rospkg import rospack
import struct
import pickle
import time
import errno
import sys

ros_slam_path = "/opt/ros-ort"
# sys.path.insert(0, ros_slam_path+"/src/frcnn/src/py-faster-rcnn")
sys.path.insert(0, ros_slam_path+"/src/frcnn/src/py-faster-rcnn/caffe-fast-rcnn/python")
sys.path.insert(0, ros_slam_path+"/src/frcnn/src/py-faster-rcnn/lib")
sys.path.insert(0, ros_slam_path+"/devel/lib/python2.7/dist-packages")
sys.path.insert(0, "/opt/ros/kinetic/lib/python2.7/dist-packages")

import rospy
# from std_msgs.msg import String
from ort_msgs.msg import objectBBMsg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rospy.numpy_msg import numpy_msg

from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
# from numpy import uint8
# import scipy.io as sio
import caffe, os, cv2
import argparse
from threading import Thread, Lock
# from multiprocessing import Lock, Process

print("imports done!")
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
#
NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}
#

# DETECTION_RUNNING = False
DETECT_RUNNING = False
detect_running = Lock()
# VIS_RUNNING = False
current_scores = []
current_boxes = []
current_frame = None
current_frame_id = None
frames_detected = 0
detection_start = time.time()

# min_c = 255
# max_c = 0
CONF_THRESH = 0.2
NMS_THRESH = 0.1

def pub_detections(pub, class_name, dets, thresh=0.5):
    global current_frame_id

    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    print("Publishing detection of class " + str(class_name))

    frameId = 0
    isKeyFrame = False

    class_name = str(class_name)

    highscorebb = None
    highscore = 0
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > highscore:
            highscorebb = bbox
            highscore = score

    bbMsg = objectBBMsg(frameId, isKeyFrame, highscorebb, class_name, highscore)
    print("publishing bb" + str(bbMsg))
    pub.publish(bbMsg)

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    print("Visualizing detection of class " + str(class_name))

    # switch red and blue
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
    # plt.draw()
    plt.savefig("kf_"+str(current_kf_id) + "_" + str(class_name) + ".png")
    print("image drawn")



def frame_detect(net, im):
    """Detect object classes in an image using pre-computed object proposals."""
    global current_scores, current_boxes

    # print("starting object detection")
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    current_scores, current_boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, current_boxes.shape[0])

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

def fake_detect(fname="pics/kf_20542.png",net=None):
    global NEW_DETECTION, current_kf, current_kf_id
    NEW_DETECTION = True
    try:
        open(fname)
    except OSError as e:
        if e.errno == errno.ENOENT:
            print("File not found")
            exit(1)
        else:
            raise
    current_kf = cv2.imread(fname)
    #
    current_kf_id = "fake_detect_frame"
    if net is not None:
        frame_detect(net)

def deserialize_and_detect_thread(msg, net):
    global current_frame, current_frame_id, DETECT_RUNNING, frames_detected, detection_start, detect_running
    im_id = msg.header.seq
    if not DETECT_RUNNING:
        DETECT_RUNNING = True
        # detect_running.acquire(1)
        # try:
        print("Starting detection of frame {}.".format(im_id))
        frames_detected += 1
        # time.sleep(1)

        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, msg.encoding)
        img = np.asarray(cv_image)
        if len(img.shape) == 2:
            img = np.asarray([img, img, img])
            img = np.swapaxes(img, 0, 2)
            img = np.swapaxes(img, 1, 0)
        current_frame = img
        current_frame_id = im_id
        if net is not None:
            frame_detect(net, img)
        # # plt.show()
        cv2.imwrite("output/f_" + str(current_frame_id) + ".png", current_frame)
        now = time.time()
        detection_time = now - detection_start
        fps = frames_detected / detection_time
        print("Running for {} sec., detection with {} fps.".format(detection_time, fps))
        # finally:
        #     detect_running.release()
        DETECT_RUNNING = False
    else:
        print("SKipping detection in frame {}".format(im_id))

def cb_frame_rec(msg, net=None):
    # global current_frame, current_frame_id, DETECT_RUNNING, frames_detected, detection_start, detect_running
    im_id = msg.header.seq
    print("Frame {} received".format(im_id))
    t = Thread(target=deserialize_and_detect_thread, args=(msg, net))
    t.start()

if __name__ == '__main__':

    rospy.init_node("frcnn")
    print("node initialized")
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()
    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))
    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id

    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    # net = None

    print '\n\nLoaded network {:s}'.format(caffemodel)
    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    detection_start = time.time()
    sub_frames = rospy.Subscriber("/image_raw", Image, cb_frame_rec, queue_size=1, callback_args=net)
    rospy.spin()

    # bb_pub = rospy.Publisher('frcnn/bb', objectBBMsg, queue_size=1)
    #
    # # fake_detect(net=net)
    #
    # ctr = 0
    # # if True:
    # while True:
    #     # Visualize detections for each class
    #     time.sleep(0.5)
    #     if not DETECT_RUNNING:
    #         # VIS_RUNNING = True
    #         for cls_ind, cls in enumerate(CLASSES[1:]):
    #             cls_ind += 1  # because we skipped background
    #             cls_boxes = current_boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
    #             cls_scores = current_scores[:, cls_ind]
    #             dets = np.hstack((cls_boxes,
    #                               cls_scores[:, np.newaxis])).astype(np.float32)
    #             keep = nms(dets, NMS_THRESH)
    #             dets = dets[keep, :]
    #             pub_detections(bb_pub, cls, dets, thresh=CONF_THRESH)
    #             # vis_detections(current_kf, cls, dets, thresh=CONF_THRESH)
    #             # break
    #         ctr += 1
    #         NEW_DETECTION = False
            # VIS_RUNNING = False

    # plt.show()
