import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rospy
from ort_msgs.msg import Object_bb_list
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sklearn.cluster import AffinityPropagation
from math import isnan
import dlib


class DlibTracker:

    def cb_bb_img_rec(self, msg):
        self.last_detected_frame_timestamp = int(msg.header.stamp.nsecs)
        print("Frame wth timestamp {} received".format(self.last_detected_frame_timestamp))
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, msg.encoding)
        img = np.asarray(cv_image)
        if len(img.shape) == 2:
            img = np.asarray([img, img, img])
            img = np.swapaxes(img, 0, 2)
            img = np.swapaxes(img, 1, 0)
        self.last_detected_frame = img
        self.last_detected_img_msg = msg

    def xywh_box_to_xyxy_box(self, bb):
        """Converts from center_x, center_y, width, height to ul_x, ul_y, lr_x, lr_y"""
        """(ul_x, ul_y) are (0,0) in the top left corner"""
        x = bb[0]
        y = bb[1]
        w = bb[2]
        h = bb[3]
        return np.array([x-(w/2.), y-(h/2.), x+(w/2.), y+(h/2.)])

    def xyxy_box_to_xywh_box(self, bb):
        """Converts from center_x, center_y, width, height to ul_x, ul_y, lr_x, lr_y"""
        """(ul_x, ul_y) are (0,0) in the top left corner"""
        x1 = bb[0]
        y1 = bb[1]
        x2 = bb[2]
        y2 = bb[3]
        return np.array([x1 + ((x2-x1) / 2), y1 + ((y2-y1) / 2), (x2-x1), (y2-y1)])

    def cb_bb_rec(self, msg):

        current_frame_bb_timestamp = int(msg.frame_timestamp)
        print("BB for frame wth timestamp {} received".format(current_frame_bb_timestamp))
        if current_frame_bb_timestamp != self.last_detected_frame_timestamp:
            if (current_frame_bb_timestamp is not None) and (self.last_detected_frame_timestamp is not None):
                print("Warning frame timestamps don't match, i.e. bounding boxes are not for this image!!!")
            return
        self.current_bbs = self.bb_msg_to_bb_dict(msg)
        self.cluster_bbs()

        bbs = self.last_clustered_bbs
        for class_name in bbs:
            for i, label in enumerate(bbs[class_name]["labels"]):
                bbox = bbs[class_name]["bboxes"][i]
                score = bbs[class_name]["scores"][i]
                if len(bbox) < 4:
                    print ("warning, bbox not set")

                drbbox = self.xywh_box_to_xyxy_box(bbox)
                if len(self.trackers) > 0:
                    return
                self.tracker_info[label] = {"bb": bbox, "score": score, "cls": class_name}
                if label not in self.trackers.keys():
                    self.trackers[label] = dlib.correlation_tracker()
                    drectangle = dlib.rectangle(int(drbbox[0]), int(drbbox[1]), int(drbbox[2]), int(drbbox[3]))
                    self.trackers[label].start_track(self.last_detected_frame, drectangle)

        # self.vis_tracking(self.last_detected_bb_clusters, write_img=False)

    def cb_camera_raw(self, msg):
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, msg.encoding)
        img = np.asarray(cv_image)
        if len(img.shape) == 2:
            img = np.asarray([img, img, img])
            img = np.swapaxes(img, 0, 2)
            img = np.swapaxes(img, 1, 0)

        current_tracked_bbs = {}

        for object_id,t in self.trackers.items():
            t.update(img)
            bb = t.get_position()
            bbox = self.xyxy_box_to_xywh_box([bb.left(), bb.top(), bb.right(), bb.bottom()])
            cls = self.tracker_info[object_id]["cls"]
            score = self.tracker_info[object_id]["score"]
            self.tracker_info[object_id]["bb"] = bbox
            if cls not in current_tracked_bbs.keys():
                current_tracked_bbs[cls] = {}
                current_tracked_bbs[cls]["labels"] = []
                current_tracked_bbs[cls]["bboxes"] = []
                current_tracked_bbs[cls]["scores"] = []

            current_tracked_bbs[cls]["labels"].append(object_id)
            current_tracked_bbs[cls]["bboxes"].append(bbox)
            current_tracked_bbs[cls]["scores"].append(score)

        if len(self.trackers) > 0:
            self.vis_tracking(current_tracked_bbs, write_img=False)


    def vis_tracking(self, bbs, write_img=False):
        im = self.last_detected_frame
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')
        for class_name in bbs:
            for i, label in enumerate(bbs[class_name]["labels"]):
                bbox = bbs[class_name]["bboxes"][i]
                score = bbs[class_name]["scores"][i]
                if len(bbox) < 4:
                    print ("warning, bbox not set")
                ax.add_patch(
                    plt.Rectangle((bbox[0], bbox[1]),
                                  bbox[2] - bbox[0],
                                  bbox[3] - bbox[1], fill=False,
                                  edgecolor='red', linewidth=3.5)
                )
                ax.text(bbox[0], bbox[1] - 2,
                        '{:s} - {:s} - {:.3f}'.format(class_name, label, score),
                        bbox=dict(facecolor='blue', alpha=0.5),
                        fontsize=14, color='white')

        plt.axis('off')
        plt.tight_layout()
        if write_img:
            plt.savefig("output/frame_" + str(self.last_detected_frame_timestamp) + ".png")
        img = self.fig_to_img_msg(fig)
        self.bb_img_pub.publish(img)

    def fig_to_img_msg(self, fig):
        # Transform figure into numpy array
        # If we haven't already shown or saved the plot, then we need to
        # draw the figure first...
        fig.canvas.draw()
        # Now we can save it to a numpy array.
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        cv_image = data
        bridge = CvBridge()
        img = bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        img.header = self.last_detected_img_msg.header

        return img

    def bb_msg_to_bb_dict(self, bb_msg):
        bb_dict = {}
        for i, obj_label in enumerate(bb_msg.object_id):
            class_name = bb_msg.class_name[i]
            if class_name not in bb_dict.keys():
                bb_dict[class_name] = {"bboxes": [], "scores": [], "labels": [], "classes": []}
            bbox = [bb_msg.bb_x[i], bb_msg.bb_y[i], bb_msg.bb_width[i], bb_msg.bb_height[i]]
            score = bb_msg.score[i]
            bb_dict[class_name]["bboxes"].append(bbox)
            bb_dict[class_name]["scores"].append(score)
            bb_dict[class_name]["labels"].append(class_name+"_"+str(i))
            bb_dict[class_name]["classes"].append(class_name)
        return bb_dict

    def cluster_bbs(self):
        self.last_clustered_bbs = {}

        # Separate by classes and cluster:
        for cls, bbs_scores in self.current_bbs.items():
            num_cls_bbs = len(bbs_scores)
            if num_cls_bbs == 0:
                return
            bbs = np.asarray(bbs_scores["bboxes"])
            af = AffinityPropagation(preference=-15000)
            af.fit(bbs)
            clustered_bb_labels = {}
            for idx, l in enumerate(af.labels_):
                if isnan(l):
                    l == 0
                l = cls+"_"+str(l)
                if l not in clustered_bb_labels.keys():
                    clustered_bb_labels[l] = []

                clustered_bb_labels[l].append(idx)
            if len(af.cluster_centers_.shape) == 2:
                clustered_bbs = af.cluster_centers_
            else:
                # Wrong array shape happens by AffinityPropagation if only one occurrence for a given class is found.
                clustered_bbs = af.cluster_centers_
                clustered_bbs = np.reshape(clustered_bbs, (1, 4))
            if len(clustered_bbs) > 0:
                if cls not in self.last_clustered_bbs.keys():
                    self.last_clustered_bbs[cls] = {}
                # TODO: This should be fixed, labels may not be ordered wrt. bbs.
                bb_labels = np.asarray(clustered_bb_labels.keys())
                bb_scores = np.ones(shape=bb_labels.shape)
                self.last_clustered_bbs[cls]["labels"] = bb_labels
                self.last_clustered_bbs[cls]["bboxes"] = clustered_bbs
                self.last_clustered_bbs[cls]["scores"] = bb_scores

    def __init__(self, max_age=1, min_hits=3):
        """
        Sets key parameters for SORT
        """
        print("Initializing the Tracker")
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.total_time = 0.0
        self.total_frames = 0
        self.last_detected_frame = None
        self.last_detected_frame_timestamp = None
        self.last_detected_img_msg = None
        self.current_bbs = {}
        self.last_clustered_bbs = {}

        self.trackers = {}
        self.tracker_info = {}
        self.cv_bridge = CvBridge()

        rospy.init_node("frcnn_tracker")
        # Subscribe to bb and image
        self.sub_bb = rospy.Subscriber("/frcnn/bb", Object_bb_list, self.cb_bb_rec, queue_size=1)
        self.sub_bb_img = rospy.Subscriber("/frcnn/bb_img", Image, self.cb_bb_img_rec, queue_size=1)
        self.sub_camera_raw = rospy.Subscriber("/frcnn_input/image_raw", Image, self.cb_camera_raw, queue_size=1)
        self.bb_img_pub = rospy.Publisher('/frcnn/bb_img_tracking', Image, queue_size=1)



        rospy.spin()