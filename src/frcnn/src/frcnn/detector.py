import time
import errno
import sys


ros_slam_path = "/opt/ros-ort"
sys.path.insert(0, ros_slam_path+"/src/frcnn/src/py-faster-rcnn")
sys.path.insert(0, ros_slam_path+"/src/frcnn/src/py-faster-rcnn/caffe-fast-rcnn/python")
sys.path.insert(0, ros_slam_path+"/src/frcnn/src/py-faster-rcnn/lib")

import rospy
from ort_msgs.msg import Object_bb_list
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from utils.timer import Timer
import numpy as np
import caffe
from threading import Thread
import os


class Detector:
    DETECT_RUNNING = False

    def __init__(self, classes, prototxt_file, caffemodel_file, args, class_properties=None):

        self.classes = classes
        self.current_scores = []
        self.current_boxes = []
        self.current_frame = None
        self.current_frame_timestamp = None
        self.current_frame_header = None
        self.frames_detected = 0
        self.detection_start = time.time()
        self.args = args
        self.CONF_THRESH = args.conf_threshold

        # print ("THRESH" + str(self.CONF_THRESH))

        self.cls_score_factors = {}
        self.set_cls_score_factors(class_properties)

        rospy.init_node("frcnn_detector")
        print("node initialized")
        cfg.TEST.HAS_RPN = True  # Use RPN for proposals

        prototxt = prototxt_file
        caffemodel = caffemodel_file

        if not os.path.isfile(caffemodel):
            raise IOError(('{:s} not found.\nDid you run ./data/script/'
                           'fetch_faster_rcnn_models.sh?').format(caffemodel))

        if not os.path.isfile(prototxt):
            raise IOError(("{:s} not found.\nMaybe this model is incompatible with the "
                           "respective network you chose.").format(caffemodel))
        if args.cpu_mode:
            caffe.set_mode_cpu()
            print("Set caffe to CPU mode")
        else:
            caffe.set_mode_gpu()
            caffe.set_device(args.gpu_id)
            cfg.GPU_ID = args.gpu_id
            print("Set caffe to GPU mode, running on GPU {}".format(cfg.GPU_ID))

        self.net = caffe.Net(prototxt, caffemodel, caffe.TEST)

        print '\n\nLoaded network {:s}'.format(caffemodel)
        # Warmup on a dummy image
        im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
        for i in xrange(2):
            _, _ = im_detect(self.net, im)

        # Create bounding box publisher
        self.bb_pub = rospy.Publisher('frcnn/bb', Object_bb_list, queue_size=10)
        # self.bb_img_pub = rospy.Publisher('frcnn/bb_img', Image, queue_size=1)

        self.detection_start = time.time()
        self.sub_frames = rospy.Subscriber("/frcnn_input/image_raw", Image, self.cb_frame_rec, queue_size=10)
        rospy.spin()

    def set_cls_score_factors(self, class_properties):
        '''
        This sets the factor to multiply the score with, depending on the object property type (e.g., shape, color, class)
        :param class_properties:
        :return:
        '''
        if class_properties == None:
            return
        for prop in class_properties.keys():
            score_factor = class_properties[prop][0]
            for cls in class_properties[prop][1]:
                self.cls_score_factors[cls] = float(score_factor)

    def pub_detections(self):
        is_keyframe = False
        timestamp = self.current_frame_header.seq
        # print("Publishing bb with timestamp {}".format(timestamp))
        frame_id = self.current_frame_header.frame_id

        bb_ul_xs = []
        bb_ul_ys = []
        bb_lr_xs = []
        bb_lr_ys = []
        bb_scores = []
        obj_labels = []
        class_names = []
        for cls_ind, cls in enumerate(self.classes[1:]):
            cls_ind += 1  # because we skipped background
            cls_boxes = self.current_boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = self.current_scores[:, cls_ind]
            for i, b in enumerate(cls_boxes):
                score = cls_scores[i]
                if cls in self.cls_score_factors.keys():
                    cls_score_factor = self.cls_score_factors[cls]
                    score *= cls_score_factor
                if float(score) < float(self.CONF_THRESH):
                    continue
                b_ul_x = b[0]
                b_ul_y = b[1]
                b_lr_x = b[2]
                b_lr_y = b[3]
                bb_ul_xs.append(b_ul_x)
                bb_ul_ys.append(b_ul_y)
                bb_lr_xs.append(b_lr_x)
                bb_lr_ys.append(b_lr_y)
                bb_scores.append(score)
                obj_labels.append(cls+"_"+str(i))
                class_names.append(cls)

        bb_msg = Object_bb_list(frame_id, timestamp, is_keyframe, bb_ul_xs, bb_ul_ys, bb_lr_xs, bb_lr_ys, class_names,
                                obj_labels, bb_scores)
        self.bb_pub.publish(bb_msg)

    def frame_detect(self, net, im):
        if self.args.cpu_mode:
            caffe.set_mode_cpu()
            # print("Set caffe to CPU mode")
        else:
            caffe.set_mode_gpu()
            caffe.set_device(self.args.gpu_id)
            cfg.GPU_ID = self.args.gpu_id
            # print("Set caffe to GPU mode, running on GPU {}".format(cfg.GPU_ID))
        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()
        self.current_scores, self.current_boxes = im_detect(net, im)
        timer.toc()
        print ('Detection took {:.3f}s for '
               '{:d} object proposals').format(timer.total_time, self.current_boxes.shape[0])

    def deserialize_and_detect_thread(self, msg):
        '''
        Start object detection. Parse image message and start frame_detect
        :param msg:
        :return:
        '''
        # If detection is not already running start a new detection
        if not Detector.DETECT_RUNNING:
            Detector.DETECT_RUNNING = True
            self.current_frame_header = msg.header
            print("Starting detection of frame {}.".format(msg.header.seq))
            self.frames_detected += 1
            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(msg, msg.encoding)
            img = np.asarray(cv_image)
            if len(img.shape) == 2:
                img = np.asarray([img, img, img])
                img = np.swapaxes(img, 0, 2)
                img = np.swapaxes(img, 1, 0)
            self.current_frame = img
            assert(self.net is not None, "No network selected")
            if self.net is not None:
                self.frame_detect(self.net, img)
                self.pub_detections()

            now = time.time()
            detection_time = now - self.detection_start
            fps = self.frames_detected / detection_time
            print("Running for {} sec., detection with {} fps.".format(detection_time, fps))

            Detector.DETECT_RUNNING = False
        # Skip detection if another detection is running already
        else:
            pass

    def cb_frame_rec(self, msg):
        t = Thread(target=self.deserialize_and_detect_thread, args=[msg])
        t.start()


