import numpy as np
import matplotlib.pyplot as plt
import rospy
from ort_msgs.msg import objectBBMsg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
# from fast_rcnn.nms_wrapper import nms

class Visualizer:

    def cb_bb_img_rec(self, msg):
        img_id = msg.header.seq
        print("frame {} received".format(img_id))
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, msg.encoding)
        img = np.asarray(cv_image)
        if len(img.shape) == 2:
            img = np.asarray([img, img, img])
            img = np.swapaxes(img, 0, 2)
            img = np.swapaxes(img, 1, 0)
        self.current_frame = img
        # self.current_frame_id = img_id


    def cb_bb_rec(self, msg):
        print("bb for frame {} received".format(msg.frameId))
        self.current_bb = msg
        self.current_frame_id = msg.frameId
        # if self.current_bb.frameId != self.current_frame_id:
        #     print("Warning frame id's don't match!!!")
        #     return

        # for cls_ind, cls in enumerate(self.clases[1:]):
            # cls_ind += 1  # because we skipped background
        cls = self.current_bb.class_name
        box = self.current_bb.bb
        score = self.current_bb.score
        # dets = np.hstack((cls_boxes,
        #                   cls_scores[:, np.newaxis])).astype(np.float32)
        # keep = nms(dets, NMS_THRESH)
        # dets = dets[keep, :]
        self.vis_detections(cls, box, score, thresh=0.5)



    def __init__(self):
        print("Initializing the Visualizer")
        rospy.init_node("frcnn_visualizer")
        # Subscribe to /frcnn/bb
        # self.sub_frames = rospy.Subscriber("/image_raw", Image, self.cb_frame_rec, queue_size=1)
        self.sub_bb = rospy.Subscriber("/frcnn/bb", objectBBMsg, self.cb_bb_rec, queue_size=1)
        self.sub_bb_img = rospy.Subscriber("/frcnn/bb_img", Image, self.cb_bb_img_rec, queue_size=1)
        self.current_frame = None
        self.current_frame_id = None
        self.current_bb = None
        rospy.spin()

    def vis_detections(self,class_name, bbox, score, thresh=0.5):
        """Draw detected bounding boxes."""
        # inds = np.where(dets[:, -1] >= thresh)[0]
        # if len(inds) == 0:
        #     return

        print("Visualizing detection of class " + str(class_name))

        # switch red and blue
        im = self.current_frame[:, :, (2, 1, 0)]
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')

        # for i in inds:
        # bbox = dets[i, :4]
        # score = dets[i, -1]

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
        plt.savefig("output/kf_" + str(self.current_frame_id) + "_" + str(class_name) + ".png")
        print("image drawn")