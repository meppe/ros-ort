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

# NMS_THRESH = 0.1
CONF_THRESH = 0.6

class Detector:
    DETECT_RUNNING = False

    def __init__(self, args, models):

        self.models = models
        model_net = args.model.split("--")
        model = model_net[0]
        net = model_net[1]
        model_info = self.models[model]
        self.classes = model_info[3]
        self.current_scores = []
        self.current_boxes = []
        self.current_frame = None
        self.current_frame_timestamp = None
        self.current_frame_header = None
        self.frames_detected = 0
        self.detection_start = time.time()
        # The first frame's header secs timestamp.
        # self.start_secs = 0

        self.CONF_THRESH = 0.2
        self.NMS_THRESH = 0.1

        rospy.init_node("frcnn_detector")
        print("node initialized")
        cfg.TEST.HAS_RPN = True  # Use RPN for proposals

        models_dir = "models/" + self.models[model][0]
        model_net_dir = self.models[model][4][net][0]
        model_subdir = self.models[model][1]
        model_pt_file = self.models[model][2]
        frcnn_path = os.getcwd() + "/src/frcnn/src/py-faster-rcnn"
        prototxt = os.path.join(frcnn_path, models_dir, model_net_dir,
                                model_subdir, model_pt_file)

        caffemodel = os.path.join(frcnn_path, 'data', self.models[model][4][net][1])
        if not os.path.isfile(caffemodel):
            raise IOError(('{:s} not found.\nDid you run ./data/script/'
                           'fetch_faster_rcnn_models.sh?').format(caffemodel))

        if not os.path.isfile(prototxt):
            raise IOError(("{:s} not found.\nMaybe this model is incompatible with the "
                           "respective network you chose.").format(caffemodel))
        if args.cpu_mode:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
            caffe.set_device(args.gpu_id)
            cfg.GPU_ID = args.gpu_id

        self.net = caffe.Net(prototxt, caffemodel, caffe.TEST)

        print '\n\nLoaded network {:s}'.format(caffemodel)
        # Warmup on a dummy image
        im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
        for i in xrange(2):
            _, _ = im_detect(self.net, im)

        # Create bounding box publisher
        self.bb_pub = rospy.Publisher('frcnn/bb', Object_bb_list, queue_size=1)
        self.bb_img_pub = rospy.Publisher('frcnn/bb_img', Image, queue_size=1)

        self.detection_start = time.time()
        self.sub_frames = rospy.Subscriber("/frcnn_input/image_raw", Image, self.cb_frame_rec, queue_size=10)
        rospy.spin()

    def pub_detections(self):
        is_keyframe = False
        time = self.current_frame_header.stamp
        # timestamp = int(time.secs * 1000000000 + time.nsecs)
        timestamp = self.current_frame_header.seq
        # print("Publishing bb with timestamp {}".format(timestamp))
        frame_id = self.current_frame_header.frame_id

        bb_xs = []
        bb_ys = []
        bb_widths = []
        bb_heights = []
        bb_scores = []
        obj_labels = []
        class_names = []
        max_score = 0
        for cls_ind, cls in enumerate(self.classes[1:]):
            cls_ind += 1  # because we skipped background
            cls_boxes = self.current_boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = self.current_scores[:, cls_ind]
            for i, b in enumerate(cls_boxes):
                score = cls_scores[i]
                max_score = max(max_score, score)
                # print max_score
                if score < CONF_THRESH:
                    continue
                b_x = b[0]
                b_y = b[1]
                b_width = b[2]
                b_height = b[3]
                bb_xs.append(b_x)
                bb_ys.append(b_y)
                bb_widths.append(b_width)
                bb_heights.append(b_height)
                bb_scores.append(score)
                obj_labels.append(cls+"_"+str(i))
                class_names.append(cls)

        bb_msg = Object_bb_list(frame_id, timestamp, is_keyframe, bb_xs, bb_ys, bb_widths, bb_heights, class_names,
                                obj_labels, bb_scores)
        self.bb_pub.publish(bb_msg)

    def frame_detect(self, net, im):
        """Detect object classes in an image using pre-computed object proposals."""
        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()
        self.current_scores, self.current_boxes = im_detect(net, im)
        timer.toc()
        print ('Detection took {:.3f}s for '
               '{:d} object proposals').format(timer.total_time, self.current_boxes.shape[0])

    def deserialize_and_detect_thread(self, msg):
        if not Detector.DETECT_RUNNING:
            Detector.DETECT_RUNNING = True
            self.current_frame_header = msg.header
            print("Starting detection of frame with timestamp {}.".format(msg.header.stamp))
            self.frames_detected += 1
            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(msg, msg.encoding)
            img = np.asarray(cv_image)
            if len(img.shape) == 2:
                img = np.asarray([img, img, img])
                img = np.swapaxes(img, 0, 2)
                img = np.swapaxes(img, 1, 0)
            self.current_frame = img
            if self.net is not None:
                self.frame_detect(self.net, img)
                # re-publish image that is worked on
                self.bb_img_pub.publish(msg)
                # publish actual detections
                self.pub_detections()

            now = time.time()
            detection_time = now - self.detection_start
            fps = self.frames_detected / detection_time
            print("Running for {} sec., detection with {} fps.".format(detection_time, fps))

            Detector.DETECT_RUNNING = False
        else:
            # print("SKipping detection in frame {}".format(im_id))
            pass

    def cb_frame_rec(self, msg):
        t = Thread(target=self.deserialize_and_detect_thread, args=[msg])
        t.start()


