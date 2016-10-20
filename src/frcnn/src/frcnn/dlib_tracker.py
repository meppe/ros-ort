from frcnn.tracker import Tracker
import numpy as np

import dlib


class DlibTracker(Tracker):


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

    def do_tracking(self):
        # By default, just visualize the last detected bbs. Overwrite this function for advanced tracking!
        if self.last_detected_bb_timestamp not in self.img_stream_queue.keys():
            print ("Warning, timestamp not found in image queue!")
            return
        last_detected_frame = self.img_stream_queue[self.last_detected_bb_timestamp]
        # self.vis_tracking(last_detected_frame, self.last_detected_bbs, write_img=False)
        # self.vis_tracking(last_detected_frame, self.last_detected_bb_clusters, write_img=False)

        bbs = self.last_detected_bb_clusters
        for label, bb in bbs.items():

                bbox = bb["bbox"]
                score = bb["score"]
                cls = bb["class"]
                timestamp = bb["timestamp"]
                drbbox = self.xywh_box_to_xyxy_box(bbox)
                # if len(self.trackers) > 0:
                #     return
                self.tracker_info[label] = {"bb": bbox, "score": score, "cls": cls, "timestamp": timestamp}
                if label not in self.trackers.keys():
                    self.trackers[label] = dlib.correlation_tracker()
                    drectangle = dlib.rectangle(int(drbbox[0]), int(drbbox[1]), int(drbbox[2]), int(drbbox[3]))
                    self.trackers[label].start_track(last_detected_frame, drectangle)

        # self.vis_tracking(self.last_detected_bb_clusters, write_img=False)

    def cb_camera_raw(self, msg):
        img = self.img_msg_2_numpy_img(msg)
        timestamp = int(msg.header.stamp.nsecs)
        self.img_stream_queue[timestamp] = img

        current_tracked_bbs = {}

        for object_id, t in self.trackers.items():
            t.update(img)
            bb = t.get_position()
            bbox = self.xyxy_box_to_xywh_box([bb.left(), bb.top(), bb.right(), bb.bottom()])
            cls = self.tracker_info[object_id]["cls"]
            score = self.tracker_info[object_id]["score"]
            self.tracker_info[object_id]["bb"] = bbox
            if object_id not in current_tracked_bbs.keys():
                current_tracked_bbs[object_id] = {}
                current_tracked_bbs[object_id]["label"] = object_id
                current_tracked_bbs[object_id]["bbox"] = bbox
                current_tracked_bbs[object_id]["score"] = score
                current_tracked_bbs[object_id]["timestamp"] = timestamp
                current_tracked_bbs[object_id]["class"] = cls

        if len(self.trackers) > 0:
            self.vis_tracking(img, current_tracked_bbs, write_img=False)

    def __init__(self):
        # super(DlibTracker, self).__init__()
        self.trackers = {}
        self.tracker_info = {}
        Tracker.__init__(self)

