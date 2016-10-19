import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import rospy
from ort_msgs.msg import Object_bb_list
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sklearn.cluster import AffinityPropagation
from math import isnan


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
        timestamp = int(msg.header.stamp.nsecs)
        self.img_stream_queue[timestamp] = img

    def cb_bb_rec(self, msg):

        self.last_detected_bbs = Tracker.bb_msg_to_bb_dict(msg)

        self.last_detected_bb_timestamp = int(msg.frame_timestamp)

        self.last_detected_bb_clusters = Tracker.cluster_bbs(self.last_detected_bbs)

        self.do_tracking()

        # clean up input data queue and remove all frames at and before the last detection.

        for t in sorted(self.img_stream_queue.keys()):
            if t <= self.last_detected_bb_timestamp:
                del self.img_stream_queue[t]
            else:
                break

    def do_tracking(self):
        # By default, just visualize the last detected bbs. Overwrite this function for advanced tracking!
        if self.last_detected_bb_timestamp not in self.img_stream_queue.keys():
            print ("Warning, timestamp not found in image queue!")
            return
        last_detected_frame = self.img_stream_queue[self.last_detected_bb_timestamp]
        # self.vis_tracking(last_detected_frame, self.last_detected_bbs, write_img=False)
        self.vis_tracking(last_detected_frame, self.last_detected_bb_clusters, write_img=False)

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
            timestamp = bbs_by_cls["timestamps"][0]
            af = AffinityPropagation(preference=-20000)
            af.fit(bbs)
            clustered_bb_labels = {}
            for idx, l in enumerate(af.labels_):
                # nan as label happens by AffinityPropagation if only one occurrence for a given class is found.
                if isnan(l):
                    l = 0
                clustered_bb_labels[idx] = cls + "_" + str(l)

            if len(af.cluster_centers_.shape) == 2:
                clustered_bbs = af.cluster_centers_
            else:
                # Wrong array shape happens by AffinityPropagation if only one occurrence for a given class is found.
                clustered_bbs = af.cluster_centers_
                clustered_bbs = np.reshape(clustered_bbs, (1, 4))

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
        self.cv_bridge = CvBridge()
        self.img_stream_queue = {}

        rospy.init_node("frcnn_tracker")
        # Subscribe to bb and image
        self.sub_bb = rospy.Subscriber("/frcnn/bb", Object_bb_list, self.cb_bb_rec, queue_size=1)
        self.sub_camera_raw = rospy.Subscriber("/frcnn_input/image_raw", Image, self.cb_camera_raw, queue_size=1)
        self.bb_img_pub = rospy.Publisher('/frcnn/bb_img_tracking', Image, queue_size=1)
        rospy.spin()
