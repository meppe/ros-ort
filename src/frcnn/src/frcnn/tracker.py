import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import rospy
from ort_msgs.msg import Object_bb_list
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sklearn.cluster import AffinityPropagation, MeanShift
from math import isnan
import cv2

class Tracker:

    def img_msg_2_numpy_img(self, msg):
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, msg.encoding)
        img = np.asarray(cv_image)
        if len(img.shape) == 2:
            img = np.asarray([img, img, img])
            img = np.swapaxes(img, 0, 2)
            img = np.swapaxes(img, 1, 0)
        return img

    # @profile
    def cb_camera_raw(self, msg):
        img = self.img_msg_2_numpy_img(msg)
        timestamp = msg.header.seq
        print("Received img for frame {}. Skipped {} frame(s).".format(timestamp, timestamp - self.last_img_timestamp - 1))
        self.last_img_timestamp = timestamp
        self.img_stream_queue[timestamp] = img
        self.update_trackers(img, timestamp)
        self.vis_tracking(img, self.current_bbs)

    # This method is meant to update self.current_bbs. This method should be overwritten for more sophisticated trackers.
    # In this simple case, we just copy the clustered BBs.
    def update_trackers(self, img, timestamp):
        pass

    # @profile
    def cb_bb_rec(self, msg):

        self.last_detected_bbs = Tracker.bb_msg_to_bb_dict(msg)

        self.last_detected_bb_timestamp = int(msg.frame_timestamp)

        if self.last_detected_bb_timestamp + self.max_time_bb_behind < self.last_img_timestamp:
            print("Warning, bbs are coming {} frames after images. This is higher than the threshold "
                  "of {}".format(self.last_img_timestamp - self.last_detected_bb_timestamp, self.max_time_bb_behind))

        self.align_detections_and_trackers(self.last_detected_bbs)

        # Clean up input data queue and remove all frames at and before the last detection.
        self.img_stream_queue = {}
        self.tracker_info_history = {}

    # This is meant to align the newly detected bbs with the existing ones from the tracking.
    # This method should be overwritten for more sophisticated tracking.
    def align_detections_and_trackers(self, bbs):
        self.current_bbs = bbs
        # Just check what the closes received frame was...
        # If a frame has been received by the detector, this does not mean
        # that it was also received by the tracker. In ROS, frames may be dropped.
        for k in sorted(self.img_stream_queue.keys()):
            if self.last_detected_bb_timestamp <= k:
                print ("{} is the frame closest to the received frame {}".format(k, self.last_detected_bb_timestamp))
                print("Difference is {}".format(abs(k - self.last_detected_bb_timestamp)))
                return

    # @profile
    def vis_tracking(self, im, bbs):
        # draw grid
        # factor = 100
        # for y in range((im.shape[0] / factor) + 1):
        #     y = y * factor
        #     im = cv2.line(im, (0,y), (im.shape[1]-1, y), (255,0,0), 2)
        #
        # for x in range((im.shape[1] / factor) + 1):
        #     x = x * factor
        #     im = cv2.line(im, (x,0), (x, im.shape[0]-1 ), (255,0,0), 2)

        for obj_id, bb in bbs.items():
            bbox = bb["bbox"]
            ul = (bbox[0], bbox[1])
            lr = (bbox[2], bbox[3])
            rect_color = (0, 0, 255)
            font_color = (255, 255, 255)
            im = cv2.rectangle(im, ul, lr, rect_color, 2)
            bbox_text = []
            bbox_text.append("{:s}".format("obj_" + str(obj_id)))
            for cls in sorted(bb["classes"], key=bb["classes"].get):
                scr = bb["classes"][cls]
                bbox_text.append("{} -- {:.2f}".format(cls, scr))
            font_height = 12
            for i, txt in enumerate(bbox_text):
                txt_ul = (ul[0], ul[1] - ((len(bbox_text) - i) * font_height))
                im = cv2.putText(im, txt, txt_ul, cv2.FONT_HERSHEY_SIMPLEX, 0.4, font_color, 1, cv2.LINE_AA)

        img_msg = self.cv_bridge.cv2_to_imgmsg(im, encoding="bgr8")

        self.bb_img_pub.publish(img_msg)

    @staticmethod
    def fig_to_img_msg(fig, cv_bridge=None, header=None):
        # Transform figure into numpy array
        # If we haven't already shown or saved the plot, then we need to
        # draw the figure first...
        if not cv_bridge:
            cv_bridge = CvBridge()
        fig.canvas.draw()
        # Now we can save it to a numpy array.
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        cv_image = data
        img = cv_bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        if header is not None:
            img.header = header
        return img

    @staticmethod
    def bb_msg_to_bb_dict(bb_msg):
        bb_dict = {}
        for i, obj_label in enumerate(bb_msg.object_id):
            if obj_label not in bb_dict.keys():
                bb_dict[obj_label] = {}
            bbox = [int(bb_msg.bb_ul_x[i]), int(bb_msg.bb_ul_y[i]), int(bb_msg.bb_lr_x[i]), int(bb_msg.bb_lr_y[i])]
            score = bb_msg.score[i]
            class_name = bb_msg.class_name[i]
            bb_dict[obj_label]["bbox"] = bbox
            bb_dict[obj_label]["score"] = score
            bb_dict[obj_label]["label"] = obj_label
            bb_dict[obj_label]["class"] = class_name
            bb_dict[obj_label]["classes"] = {class_name: score}
            bb_dict[obj_label]["timestamp"] = int(bb_msg.frame_timestamp)
        return bb_dict

    @staticmethod
    def cluster_bbs(bbs):

        bbs_by_classes = {}
        # Separate by classes first:
        for object_ids, bb in bbs.items():
            cls = bb["class"]
            if cls not in bbs_by_classes.keys():
                bbs_by_classes[cls] = {"bboxes": [], "scores": [], "labels": [], "classes": [], "timestamps": []}
            bbs_by_classes[cls]["bboxes"].append(bb["bbox"])
            bbs_by_classes[cls]["scores"].append(bb["score"])
            bbs_by_classes[cls]["labels"].append(bb["label"])
            bbs_by_classes[cls]["classes"].append(bb["class"])
            bbs_by_classes[cls]["timestamps"].append(bb["timestamp"])

        # Now cluster by classes
        bb_clusters_by_id = {}
        for cls, bbs_by_cls in bbs_by_classes.items():
            num_cls_bbs = len(bbs_by_cls)
            if num_cls_bbs == 0:
                return
            bbs = np.asarray(bbs_by_cls["bboxes"])
            print("Number of bbs for {}: {}".format(cls, str(len(bbs))))
            timestamp = bbs_by_cls["timestamps"][0]
            clustering_algo = AffinityPropagation(preference=-12000)
            # clustering_algo = MeanShift()
            clustering_algo.fit(bbs)

            clustered_bb_labels = {}
            for idx, l in enumerate(clustering_algo.labels_):
                clustered_bbs = clustering_algo.cluster_centers_
                # nan as label happens with AffinityPropagation if only one occurrence for a given class is found. In that case, also the shape of the bbs is wrong.
                if isnan(l):
                    l = 0
                clustered_bb_labels[idx] = cls + "_" + str(l)

            # Often, if the above nan case holds, the shape of the bbs are also wrong. This is corrected below:
            if len(clustering_algo.cluster_centers_.shape) == 3:
                n_bbs = clustered_bbs.shape[1]
                clustered_bbs = np.reshape(clustered_bbs, (n_bbs, 4))

            # Now sort clusters by object label again.
            for idx, bb in enumerate(clustered_bbs):
                obj_id = clustered_bb_labels[idx]
                bb_clusters_by_id[obj_id] = {}
                bb_clusters_by_id[obj_id]["bbox"] = bb
                bb_clusters_by_id[obj_id]["score"] = 1
                bb_clusters_by_id[obj_id]["label"] = obj_id
                bb_clusters_by_id[obj_id]["class"] = cls
                bb_clusters_by_id[obj_id]["timestamp"] = timestamp

        return bb_clusters_by_id

    def __init__(self):
        print("Initializing the Tracker")
        self.last_detected_bbs = {}
        self.last_detected_bb_clusters = {}
        self.last_detected_bb_timestamp = None
        self.current_bbs = {}
        self.cv_bridge = CvBridge()
        self.img_stream_queue = {}
        self.tracker_info_history = {}
        # self.bb_history = {}

        # Maximum amout of time (or frames) that the bb message comes after the image. This is used to clean up the
        # image queue.
        self.max_time_bb_behind = 20

        rospy.init_node("frcnn_tracker")
        # Subscribe to bb and image
        self.sub_bb = rospy.Subscriber("/frcnn/bb", Object_bb_list, self.cb_bb_rec, queue_size=10)
        self.sub_camera_raw = rospy.Subscriber("/frcnn_input/image_raw", Image, self.cb_camera_raw, queue_size=10)
        # This subscribes to the images that are actually processed by the detector. These are also stacked on the
        # image queue to assure that trackers are started with frames on which objectes were detected
        # self.sub_camera_raw_detection = rospy.Subscriber("/frcnn/bb_img", Image, self.cb_camera_raw, queue_size=10)
        self.bb_img_pub = rospy.Publisher('/frcnn/bb_img_tracking', Image, queue_size=1)
        self.last_img_timestamp = 0
        rospy.spin()
