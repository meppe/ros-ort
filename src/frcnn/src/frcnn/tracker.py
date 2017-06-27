import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import rospy
from ort_msgs.msg import Object_bb_list
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sklearn.cluster import AffinityPropagation, MeanShift
from math import isnan
import cv2
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

    # @profile
    def cb_camera_raw(self, msg):
        img = self.img_msg_2_numpy_img(msg)
        timestamp = msg.header.seq
        skip = timestamp - self.last_img_timestamp - 1
        if skip > 0:
            print("Received img for frame {}. Skipped {} frame(s). Frames coming too fast or system too slow.".format(
                timestamp, skip))
        self.last_img_timestamp = timestamp
        self.img_stream_queue[timestamp] = img
        self.update_trackers(img, timestamp)
        self.vis_tracking(img, self.current_bbs)

    # This method is meant to update self.current_bbs. This method should be overwritten for more sophisticated trackers.
    def update_trackers(self, img, timestamp):
        pass

    # @profile
    def cb_bb_rec(self, msg):

        self.last_detected_bbs = Tracker.bb_msg_to_bb_dict(msg)
        self.last_detected_bb_timestamp = int(msg.frame_timestamp)

        print("Received BB object detections for frame {}. Last image frame is {}.".format(
            self.last_detected_bb_timestamp, self.last_img_timestamp))

        self.align_detections_and_trackers(self.last_detected_bbs)

        # Clean up input data queue and remove all frames at and before the last detection.
        # img_queue_to_keep = {}
        imgs_to_del = []
        tracker_history_to_keep = {}
        for timestamp in sorted(self.img_stream_queue.keys()):
            if timestamp >= self.last_detected_bb_timestamp and timestamp < self.last_img_timestamp:
                # Wait for tracking to be finished for that timestamp.
                # while timestamp not in self.tracker_info_history.keys():
                #     print("Waiting for tracking frame {} to finish.".format(str(timestamp)))
                #     time.sleep(0.01)
                # if timestamp not in self.tracker_info_history.keys():
                #     print("Warning, frame with timestamp {} not in tracker info history.".format(str(timestamp)))
                #     continue
                imgs_to_del.append(timestamp)
                # img_queue_to_keep[timestamp] = self.img_stream_queue[timestamp]
                # tracker_history_to_keep[timestamp] = self.tracker_info_history[timestamp]

        for img in imgs_to_del:
            del self.img_stream_queue[img]

    def cb_txt_interface(self, interface_string_msg):
        print "Received classes string message: {}".format(str(interface_string_msg))
        interface_string = str(interface_string_msg).replace(" ", "")
        interface_string = interface_string[5:] # The first 5 letters are "data:"
        if interface_string == "all":
            self.classes_to_display = []
            print("Displaying all classes.")
        elif interface_string == "mask":
            # Toggle masking
            self.mask_objects = not self.mask_objects
            print("Setting masking to {}".format(self.mask_objects))
        elif interface_string == "single":
            # Toggle whether to display only the one single object with the highest score.
            self.single_object = not self.single_object
            print("Setting masking to {}".format(self.mask_objects))
        elif interface_string[:5] == "file:":
            fname = interface_string[5:]
            print ("Setting file to write to {} ".format(fname))
            self.write_to_file = fname
        else:
            self.classes_to_display = interface_string.split(",")

    # This is meant to align the newly detected bbs with the existing ones from the tracking.
    # This method should be overwritten for more sophisticated tracking.
    def align_detections_and_trackers(self, bbs):
        self.current_bbs = bbs
        # Just check what the closes received frame was...
        # If a frame has been received by the detector, this does not mean
        # that it was also received by the tracker. In ROS, frames may be dropped.
        for k in sorted(self.img_stream_queue.keys()):
            if self.last_detected_bb_timestamp <= k:
                print ("Frame {} is the frame closest to the received BB frame {}. Skipping {} frames.".format(
                    k, self.last_detected_bb_timestamp, abs(k - self.last_detected_bb_timestamp)))
                return

    # @profile
    def vis_tracking(self, im, bbs):
        '''

        :param im: The image to display.
        :param bbs: The bounding boxes with object id and class information to display.
        :param classes: If a non-empty list is given, then only display object classes in the list.
        :param bg_color: If not None, then replace all background around the bounding boxes with this color, specified as RGB triple.
        :param write_to_file: If a file path is provided, then write image as file to disk.
        :param display_labels: If false, do not display labels around boxes.
        :return:
        '''

        # draw grid
        # factor = 100
        # for y in range((im.shape[0] / factor) + 1):
        #     y = y * factor
        #     im = cv2.line(im, (0,y), (im.shape[1]-1, y), (255,0,0), 2)
        #
        # for x in range((im.shape[1] / factor) + 1):
        #     x = x * factor
        #     im = cv2.line(im, (x,0), (x, im.shape[0]-1 ), (255,0,0), 2)
        # print("Image shape: {}".format(im.shape))
        rect_color = (0, 0, 255)
        font_color = (255, 255, 255)
        bg_color = (255, 0, 0)
        mask_color = (1, 1, 1)
        width = im.shape[1]
        height = im.shape[0]
        depth = 3
        obj_boxes_mask = np.zeros(shape=(height, width, depth), dtype=np.uint8)

        # Compute object socres
        bb_scores = {}
        for obj_id, bb in bbs.items():
            bb_scores[obj_id] = 0
            for cls in sorted(bb["classes"], key=bb["classes"].get, reverse=True):
                scr = float(bb["classes"][cls])
                if self.classes_to_display == [] or cls in self.classes_to_display:
                    bb_scores[obj_id] += scr
        max_score_obj = max(bb_scores, key=bb_scores.get)

        for obj_id, bb in bbs.items():
            # If the goal is to show only a single object, and if this is not the object with the highest score, then continue.
            if self.single_object and obj_id != max_score_obj:
                continue
            # Throw away those classes with a low score and
            # throw away those bbs that do not show one of the classes to display
            bb_orig = bb
            for cls in sorted(bb["classes"], key=bb["classes"].get, reverse=True):
                scr = float(bb["classes"][cls])
                # Score has to be above a certain threshold
                if scr < self.class_threshold:
                    del bb_orig["classes"][cls]
                    continue
                # Class has to be in classes_to_display
                if self.classes_to_display != [] and cls not in self.classes_to_display:
                    del bb_orig["classes"][cls]
                    continue

            bb = bb_orig

            # Throw away those bbs with a low score (the total score without removing classes)
            if bb["score"] < self.cum_threshold:
                print("Score for bounding box for object {} too low to display ({})".format(obj_id, bb["score"]))
                continue

            if len(bb["classes"].keys()) == 0:
                continue

            bbox = bb["bbox"]
            ul = (bbox[0], bbox[1])
            lr = (bbox[2], bbox[3])

            # Generate object box masks
            bbox_arr = np.array([[ul[0], ul[1]], [ul[0], lr[1]], [lr[0], lr[1]], [lr[0], ul[1]]], dtype=np.int32)
            cv2.fillPoly(obj_boxes_mask, [bbox_arr], mask_color)

            # Draw rectangle and write label if not masking objects
            if not self.mask_objects:
                # Rectangle
                im = cv2.rectangle(im, ul, lr, rect_color, 2)
                # Labels
                bbox_text = []
                bbox_text.append("{:s} -- {}".format("obj_" + str(obj_id), bb["score"]))
                for cls in sorted(bb["classes"], key=bb["classes"].get, reverse=True):
                    scr = bb["classes"][cls]
                    bbox_text.append("{} -- {:.2f}".format(cls, scr))
                font_height = 12
                txt_ul = [ul[0], ul[1] - (len(bbox_text) * font_height)]
                txt_ul[0] = max(txt_ul[0], 0)
                txt_ul[1] = max(txt_ul[1], font_height)
                for i, txt in enumerate(bbox_text):
                    line_ul = (txt_ul[0], txt_ul[1] + font_height * i)
                    im = cv2.putText(im, txt, line_ul, cv2.FONT_HERSHEY_SIMPLEX, 0.4, font_color, 1, cv2.LINE_AA)

        if self.mask_objects:
            np.clip(obj_boxes_mask, 0, 1, out=obj_boxes_mask)
            im *= obj_boxes_mask
            inverse_mask = (obj_boxes_mask - 1) * (-1)
            inverse_mask *= bg_color
            inverse_mask = np.uint8(inverse_mask)
            im += inverse_mask

        # Write file to disk if self.write_to_file is set.
        if self.write_to_file != '':
            print("Writing image to {}".format(self.write_to_file))
            cv2.imwrite(self.write_to_file, im)

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

    def __init__(self, mask_objects=False, single_object=False, classes_to_display=[], write_to_file=''):
        print("Initializing the Tracker")
        self.last_detected_bbs = {}
        self.last_detected_bb_clusters = {}
        self.last_detected_bb_timestamp = None
        self.current_bbs = {}
        self.cv_bridge = CvBridge()
        self.img_stream_queue = {}
        self.tracker_info_history = {}

        # Maximum amout of time (or frames) that the bb message may be received after the image. This is used to clean up the image queue.
        self.max_time_bb_behind = 20

        rospy.init_node("frcnn_tracker")
        # Subscribe to bb and image and txt console messages
        self.sub_bb = rospy.Subscriber("/frcnn/bb", Object_bb_list, self.cb_bb_rec, queue_size=10)
        self.sub_camera_raw = rospy.Subscriber("/frcnn_input/image_raw", Image, self.cb_camera_raw, queue_size=10)
        self.sub_bb = rospy.Subscriber("/frcnn/interface_input", String, self.cb_txt_interface, queue_size=10)
        # Publish img with bounding boxes
        self.bb_img_pub = rospy.Publisher('/frcnn/bb_img_tracking', Image, queue_size=10)
        self.last_img_timestamp = 0
        self.mask_objects = mask_objects
        self.single_object = single_object
        self.classes_to_display = classes_to_display
        self.write_to_file = write_to_file
        rospy.spin()
