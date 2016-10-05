import numpy as np
import matplotlib.pyplot as plt
import rospy
from ort_msgs.msg import Object_bb_list
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class Visualizer:

    def cb_bb_img_rec(self, msg):
        self.current_frame_img_timestamp = int(msg.header.stamp.nsecs)
        print("Frame wth timestamp {} received".format(self.current_frame_img_timestamp))
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, msg.encoding)
        img = np.asarray(cv_image)
        if len(img.shape) == 2:
            img = np.asarray([img, img, img])
            img = np.swapaxes(img, 0, 2)
            img = np.swapaxes(img, 1, 0)
        self.current_frame = img

    def cb_bb_rec(self, msg):
        current_frame_bb_timestamp = int(msg.frame_timestamp)
        print("BB for frame wth timestamp {} received".format(current_frame_bb_timestamp))
        self.current_bbs = msg
        if current_frame_bb_timestamp != self.current_frame_img_timestamp:
            if (current_frame_bb_timestamp is not None) and (self.current_frame_img_timestamp is not None):
                print("Warning frame timestamps don't match, i.e. bounding boxes are not for this image!!!")
            return

        self.vis_detections(msg)

    def vis_detections(self, bb_msg):
        """Draw detected bounding boxes."""
        if self.current_frame is None:
            return
        num_detections = len(bb_msg.object_id)
        if num_detections == 0:
            return
        # switch red and blue, then draw img
        im = self.current_frame[:, :, (2, 1, 0)]
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')
        # Add boxes, scores and class names
        for i, obj_label in enumerate(bb_msg.object_id):
            class_name = bb_msg.class_name[i]
            bbox = [bb_msg.bb_x[i], bb_msg.bb_y[i], bb_msg.bb_width[i], bb_msg.bb_height[i]]
            score = bb_msg.score[i]
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

        ax.set_title('{} detections above threshold found'.format(num_detections), fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("output/kf_" + str(self.current_frame_img_timestamp) + ".png")

    def __init__(self):
        print("Initializing the Visualizer")
        rospy.init_node("frcnn_visualizer")
        # Subscribe to bb and image
        self.sub_bb = rospy.Subscriber("/frcnn/bb", Object_bb_list, self.cb_bb_rec, queue_size=1)
        self.sub_bb_img = rospy.Subscriber("/frcnn/bb_img", Image, self.cb_bb_img_rec, queue_size=1)
        self.current_frame = None
        self.current_frame_img_timestamp = None
        self.current_bbs = None
        rospy.spin()