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
import time

class Tracker:

    def img_msg_2_numpy_img(self, msg):
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, msg.encoding)
        img = np.asarray(cv_image)
        if len(img.shape) == 2:
            img = np.asarray([img, img, img])
            img = np.swapaxes(img, 0, 2)
            img = np.swapaxes(img, 1, 0)
        return img

    def cb_camera_raw(self, msg):
        img = self.img_msg_2_numpy_img(msg)
        # time = msg.header.stamp
        # timestamp = int(time.secs * 1000000000 + time.nsecs)
        timestamp = msg.header.seq
        self.img_stream_queue[timestamp] = img
        self.update_trackers(img, timestamp)
        self.vis_tracking(img, self.current_bbs, write_img=False)

    # This method is meant to update self.current_bbs. This method should be overwritten for more sophisticated trackers.
    # In this simple case, we just copy the clustered BBs.
    def update_trackers(self, img, timestamp):
        self.current_bbs = self.last_detected_bbs

    def cb_bb_rec(self, msg):

        self.last_detected_bbs = Tracker.bb_msg_to_bb_dict(msg)

        self.last_detected_bb_timestamp = int(msg.frame_timestamp)

        # self.last_detected_bb_clusters = Tracker.cluster_bbs(self.last_detected_bbs)

        self.align_detections_and_trackers(self.last_detected_bbs)

        # clean up input data queue and remove all frames at and before the last detection.

        print("Deleting frames up to {}.".format(self.last_detected_bb_timestamp))
        for t in sorted(self.img_stream_queue.keys()):
            if t <= self.last_detected_bb_timestamp:
                del self.img_stream_queue[t]
                # print("Deleting {}".format(t))
            else:
                break

    # This is meant to align the newly detected bbs with the existing ones from the tracking. This method should be overwritten for more sophisticated tracking.
    def align_detections_and_trackers(self, bbs):
        # By default, just visualize the last detected bbs. Overwrite this function for advanced tracking!
        for k in sorted(self.img_stream_queue.keys()):
            if self.last_detected_bb_timestamp >= k:
                print ("{} is the keyframe closest to the received keyframe {}".format(k, self.last_detected_bb_timestamp))
                print("Difference is {}".format(abs(k - self.last_detected_bb_timestamp)))
                return
        # if self.last_detected_bb_timestamp not in self.img_stream_queue.keys():
        #     print ("img queue")
        #     for k in sorted(self.img_stream_queue.keys()):
        #         print (k)
        #     print ("Frame {} not found in image queue! -- Warning!".format(self.last_detected_bb_timestamp))
        #     return

    def vis_tracking(self, im, bbs, write_img=False):
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')
        for obj_id, bb in bbs.items():
            class_name = bb["class"]
            bbox = bb["bbox"]
            score = bb["score"]
            if len(bbox) < 4:
                print ("warning, bbox not set")
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='red', linewidth=3.5)
            )
            ax.text(bbox[0], bbox[1] - 2,
                    '{:s} - {:s} - {:.3f}'.format(class_name, obj_id, score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')

        plt.axis('off')
        plt.tight_layout()
        if write_img:
            plt.savefig("output/frame_" + str(self.last_detected_bb_timestamp) + ".png")
        # img = Tracker.fig_to_img_msg(fig, self.last_detected_img_msg.header)
        img = Tracker.fig_to_img_msg(fig)
        self.bb_img_pub.publish(img)

    @staticmethod
    def fig_to_img_msg(fig, header=None):
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
        if header is not None:
            img.header = header
        return img

    @staticmethod
    def bb_msg_to_bb_dict(bb_msg):
        bb_dict = {}
        for i, obj_label in enumerate(bb_msg.object_id):
            if obj_label not in bb_dict.keys():
                bb_dict[obj_label] = {}
            bbox = [bb_msg.bb_x[i], bb_msg.bb_y[i], bb_msg.bb_width[i], bb_msg.bb_height[i]]
            score = bb_msg.score[i]
            class_name = bb_msg.class_name[i]
            bb_dict[obj_label]["bbox"] = bbox
            bb_dict[obj_label]["score"] = score
            bb_dict[obj_label]["label"] = obj_label
            bb_dict[obj_label]["class"] = class_name
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

        rospy.init_node("frcnn_tracker")
        # Subscribe to bb and image
        self.sub_bb = rospy.Subscriber("/frcnn/bb", Object_bb_list, self.cb_bb_rec, queue_size=1)
        self.sub_camera_raw = rospy.Subscriber("/frcnn_input/image_raw", Image, self.cb_camera_raw, queue_size=100)
        self.bb_img_pub = rospy.Publisher('/frcnn/bb_img_tracking', Image, queue_size=1)
        rospy.spin()
