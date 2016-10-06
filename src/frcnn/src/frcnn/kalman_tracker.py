import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rospy
from ort_msgs.msg import Object_bb_list
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sklearn.cluster import AffinityPropagation
from filterpy.kalman import KalmanFilter
from sklearn.utils.linear_assignment_ import linear_assignment
from math import isnan

class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self,bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        #define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def convert_bbox_to_z(self, bbox):
        """
        Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
          [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
          the aspect ratio
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h  # scale is just area
        r = w / h
        return np.array([x, y, s, r]).reshape((4, 1))

    def convert_x_to_bbox(self, x, score=None):
        """
        Takes a bounding box in the form [x,y,s,r] and returns it in the form
          [x1,y1,x2,x2] where x1,y1 is the top left and x2,y2 is the bottom right
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if (score == None):
            return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
        else:
            return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))

    def update(self,bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
          self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
          self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.convert_x_to_bbox(self.kf.x)

class KalmanTracker:

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
        self.current_img_msg = msg

    def cb_bb_rec(self, msg):
        current_frame_bb_timestamp = int(msg.frame_timestamp)
        print("BB for frame wth timestamp {} received".format(current_frame_bb_timestamp))
        if current_frame_bb_timestamp != self.current_frame_img_timestamp:
            if (current_frame_bb_timestamp is not None) and (self.current_frame_img_timestamp is not None):
                print("Warning frame timestamps don't match, i.e. bounding boxes are not for this image!!!")
            return
        self.current_bbs = self.bb_msg_to_bb_dict(msg)
        self.cluster_bbs()

        self.track_bbs(self.current_clustered_bbs)

        # self.vis_tracking(self.current_tracked_bbs, write_img=False)
        self.vis_tracking(self.current_clustered_bbs, write_img=False)
        # self.vis_tracking(self.current_bbs, write_img=False)

    def track_bbs(self, bbs):
        dets = []
        self.trackers = {}
        # self.current_tracked_bbs = {"bboxes" : [], "scores" : []}
        for class_name in bbs.keys():
            scores = bbs[class_name]["scores"]
            scores = np.reshape(scores, newshape=(len(scores), 1))
            bbs_scores = np.concatenate((bbs[class_name]["bboxes"],  scores), axis=1)
            dets.append(bbs_scores)
            if len(dets) == 0:
                return
            np_dets = np.asarray(dets)
            np_dets = np_dets[0]
            np_dets[:, 2:4] += np_dets[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
            self.trackers[class_name] = self.update(np_dets, class_name)
            self.current_tracked_bbs[class_name] = {"bboxes": [], "scores": [], "labels": [], "classes": []}
            for t in self.trackers[class_name]:
                t = t.astype(np.uint32)
                self.current_tracked_bbs[class_name]["bboxes"].append(t)
                self.current_tracked_bbs[class_name]["scores"].append(1.0)
                self.current_tracked_bbs[class_name]["labels"].append("label")
                self.current_tracked_bbs[class_name]["classes"].append(class_name)

    def vis_tracking(self, bbs, write_img=False):
        # im = self.current_frame[:, :, (2, 1, 0)]
        im = self.current_frame
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
            plt.savefig("output/frame_" + str(self.current_frame_img_timestamp) + ".png")
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
        img.header = self.current_img_msg.header

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
        self.current_clustered_bbs = {}

        # Separate by classes and cluster:
        for cls, bbs_scores in self.current_bbs.items():
            num_cls_bbs = len(bbs_scores)
            if num_cls_bbs == 0:
                return
            bbs = np.asarray(bbs_scores["bboxes"])
            # af = AffinityPropagation(preference=(bbs_scores["scores"]*(-5000)))
            af = AffinityPropagation(preference=-15000)
            af.fit(bbs)
            clustered_bb_labels = {}
            bb_labels = []
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
                if cls not in self.current_clustered_bbs.keys():
                    self.current_clustered_bbs[cls] = {}
                # TODO: This should be fixed, labels may not be ordered wrt. bbs.
                bb_labels = np.asarray(clustered_bb_labels.keys())
                bb_scores = np.ones(shape=bb_labels.shape)
                self.current_clustered_bbs[cls]["labels"] = bb_labels
                self.current_clustered_bbs[cls]["bboxes"] = clustered_bbs
                self.current_clustered_bbs[cls]["scores"] = bb_scores

    def convert_xywh_box_to_xyxy_box(self, bb):
        """Conversts from center_x, center_y, width, height to ul_x, ul_y, lr_x, lr_y"""
        x = bb[0]
        y = bb[1]
        w = bb[2]
        h = bb[3]
        return np.array([x-(w/2.), y+(h/2.), x+(w/2.), y-(h/2.)])

    def iou(self, bb_test, bb_gt):
        """
        Computes IUO between two bboxes in the form [x1,y1,x2,y2]
        """
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
                  + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
        return (o)

    def associate_detections_to_trackers(self, detections, trackers, iou_threshold=0.3):
        """
        Assigns detections to tracked object (both represented as bounding boxes)
        Returns 3 lists of matches, unmatched_detections and unmatched_trackers
        """
        if (len(trackers) == 0):
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self.iou(det, trk)
        matched_indices = linear_assignment(-iou_matrix)

        unmatched_detections = []
        for d, det in enumerate(detections):
            if (d not in matched_indices[:, 0]):
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if (t not in matched_indices[:, 1]):
                unmatched_trackers.append(t)

        # filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if (iou_matrix[m[0], m[1]] < iou_threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if (len(matches) == 0):
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def update(self, dets, cls):
        """
        Params:
          dets - a numpy array of detections in the format [[x,y,w,h,score],[x,y,w,h,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        if cls not in self.trackers.keys():
            self.trackers[cls] = []
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers[cls]), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[cls][t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if (np.any(np.isnan(pos))):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers[cls].pop(t)
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, trks)

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers[cls]):
            if (t not in unmatched_trks):
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets[d, :][0])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers[cls].append(trk)
        i = len(self.trackers[cls])
        for trk in reversed(self.trackers[cls]):
            d = trk.get_state()[0]
            if ((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers[cls].pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))

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
        self.current_frame = None
        self.current_frame_img_timestamp = None
        self.current_bbs = {}
        self.current_clustered_bbs = {}
        self.current_tracked_bbs = {}
        self.trackers = {}

        rospy.init_node("frcnn_tracker")
        # Subscribe to bb and image
        self.sub_bb = rospy.Subscriber("/frcnn/bb", Object_bb_list, self.cb_bb_rec, queue_size=1)
        self.sub_bb_img = rospy.Subscriber("/frcnn/bb_img", Image, self.cb_bb_img_rec, queue_size=1)
        self.bb_img_pub = rospy.Publisher('/frcnn/bb_img_tracking', Image, queue_size=1)

        rospy.spin()