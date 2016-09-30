import time
import errno
import sys


ros_slam_path = "/opt/ros-ort"
sys.path.insert(0, ros_slam_path+"/src/frcnn/src/py-faster-rcnn")
sys.path.insert(0, ros_slam_path+"/src/frcnn/src/py-faster-rcnn/caffe-fast-rcnn/python")
sys.path.insert(0, ros_slam_path+"/src/frcnn/src/py-faster-rcnn/lib")
# sys.path.insert(0, ros_slam_path+"/devel/lib/python2.7/dist-packages")
# sys.path.insert(0, "/opt/ros/kinetic/lib/python2.7/dist-packages")

import rospy
from ort_msgs.msg import objectBBMsg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import caffe, os, cv2
from threading import Thread, Lock

NMS_THRESH = 0.1
CONF_THRESH = 0.2

class Detector:
    DETECT_RUNNING = False

    def __init__(self, args, nets, classes):

        self.nets = nets
        self.clases = classes
        self.current_scores = []
        self.current_boxes = []
        self.current_frame = None
        self.current_frame_id = None
        self.frames_detected = 0
        self.detection_start = time.time()

        self.CONF_THRESH = 0.2
        self.NMS_THRESH = 0.1

        rospy.init_node("frcnn")
        print("node initialized")
        cfg.TEST.HAS_RPN = True  # Use RPN for proposals

        prototxt = os.path.join(cfg.MODELS_DIR, self.nets[args.demo_net][0],
                                'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
        caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                                  self.nets[args.demo_net][1])
        if not os.path.isfile(caffemodel):
            raise IOError(('{:s} not found.\nDid you run ./data/script/'
                           'fetch_faster_rcnn_models.sh?').format(caffemodel))
        if args.cpu_mode:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
            caffe.set_device(args.gpu_id)
            cfg.GPU_ID = args.gpu_id

        self.net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        # net = None

        print '\n\nLoaded network {:s}'.format(caffemodel)
        # Warmup on a dummy image
        im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
        for i in xrange(2):
            _, _ = im_detect(self.net, im)

        # Create bounding box publisher
        self.bb_pub = rospy.Publisher('frcnn/bb', objectBBMsg, queue_size=1)

        self.detection_start = time.time()
        # args = [self.net, self.bb_pub]
        self.sub_frames = rospy.Subscriber("/image_raw", Image, self.cb_frame_rec, queue_size=1)
        rospy.spin()

    def pub_detections(self):
        for cls_ind, cls in enumerate(self.clases[1:]):
            cls_ind += 1  # because we skipped background
            cls_boxes = self.current_boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = self.current_scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            self.pub_class_detections(cls, dets, thresh=CONF_THRESH)

    def pub_class_detections(self, class_name, dets, thresh=CONF_THRESH):
        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            return

        print("Publishing detection of class " + str(class_name))

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

        bbMsg = objectBBMsg(self.current_frame_id, isKeyFrame, highscorebb, class_name, highscore)
        print("publishing bb" + str(bbMsg))
        self.bb_pub.publish(bbMsg)

    def frame_detect(self, net, im):
        """Detect object classes in an image using pre-computed object proposals."""
        # global current_scores, current_boxes
        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()
        self.current_scores, self.current_boxes = im_detect(net, im)
        timer.toc()
        print ('Detection took {:.3f}s for '
               '{:d} object proposals').format(timer.total_time, self.current_boxes.shape[0])

    def fake_detect(self, fname="pics/kf_20542.png",net=None):
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
            self.frame_detect(net)

    def deserialize_and_detect_thread(self, msg):
        im_id = msg.header.seq
        if not Detector.DETECT_RUNNING:
            Detector.DETECT_RUNNING = True
            print("Starting detection of frame {}.".format(im_id))
            self.frames_detected += 1
            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(msg, msg.encoding)
            img = np.asarray(cv_image)
            if len(img.shape) == 2:
                img = np.asarray([img, img, img])
                img = np.swapaxes(img, 0, 2)
                img = np.swapaxes(img, 1, 0)
            self.current_frame = img
            self.current_frame_id = im_id
            if self.net is not None:
                self.frame_detect(self.net, img)
                self.pub_detections()

            cv2.imwrite("output/f_" + str(im_id) + ".png", img)
            now = time.time()
            detection_time = now - self.detection_start
            fps = self.frames_detected / detection_time
            print("Running for {} sec., detection with {} fps.".format(detection_time, fps))

            Detector.DETECT_RUNNING = False
        else:
            print("SKipping detection in frame {}".format(im_id))

    def cb_frame_rec(self, msg):
        im_id = msg.header.seq
        print("Frame {} received".format(im_id))
        # self.deserialize_and_detect_thread(msg)
        t = Thread(target=self.deserialize_and_detect_thread, args=[msg])
        t.start()


