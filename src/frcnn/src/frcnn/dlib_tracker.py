from frcnn.tracker import Tracker
import numpy as np
import time
import dlib


class DlibTracker(Tracker):

    @staticmethod
    def xywh_box_to_xyxy_box(bb):
        """Converts from center_x, center_y, width, height to ul_x, ul_y, lr_x, lr_y"""
        """(ul_x, ul_y) are (0,0) in the top left corner"""
        x = bb[0]
        y = bb[1]
        w = bb[2]
        h = bb[3]
        return np.array([x-(w/2.), y-(h/2.), x+(w/2.), y+(h/2.)]).tolist()

    @staticmethod
    def xyxy_box_to_xywh_box(bb):
        """Converts from center_x, center_y, width, height to ul_x, ul_y, lr_x, lr_y"""
        """(ul_x, ul_y) are (0,0) in the top left corner"""
        x1 = bb[0]
        y1 = bb[1]
        x2 = bb[2]
        y2 = bb[3]
        return np.array([x1 + ((x2-x1) / 2), y1 + ((y2-y1) / 2), (x2-x1), (y2-y1)]).tolist()

    @staticmethod
    def iou(bb_test, bb_gt):
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

    def align_detections_and_trackers(self, bbs):

        bb_timestamp = self.last_detected_bb_timestamp
        closest_timestamp = None
        for k in sorted(self.img_stream_queue.keys()):
            if self.last_detected_bb_timestamp <= k:
                closest_timestamp = k
                print ("{} is the frame closest to the received frame {}".format(k, self.last_detected_bb_timestamp))
                print("Difference is {}".format(abs(k - self.last_detected_bb_timestamp)))
        if closest_timestamp not in self.img_stream_queue.keys():
            print ("Warning, timestamp {} not found in image queue!".format(closest_timestamp))
            return
        if closest_timestamp not in self.tracker_info_history.keys():
            print ("Warning, timestamp {} not found in tracker history!".format(closest_timestamp))
            return

        while self.tracker_update_running == True:
            time.sleep(0.001)
        self.tracker_alignment_running = True
        tracker_info = self.tracker_info_history[closest_timestamp]
        updated_trackers = []

        for label, bb in bbs.items():
            bbox = bb["bbox"]
            bbox_xyxy = DlibTracker.xywh_box_to_xyxy_box(bbox)
            score = bb["score"]
            cls = bb["class"]
            timestamp = bb["timestamp"]

            is_new_object = True
            # Iterate through all tracker states at the timestamp of the BB and compute iou.
            # If IOU is below threshold, match it.
            ious = {}
            for obj_id, t_info in tracker_info.items():
                ious[obj_id] = DlibTracker.iou(t_info["bbox_xyxy"], bbox_xyxy)
            # Get obj_id that has the closest bb:
            if len(ious.keys()) > 0:
                obj_id = max(ious, key=ious.get)
                max_iou = ious[obj_id]
                if max_iou > self.iou_threshold:
                    is_new_object = False
                    if cls not in tracker_info[obj_id]["classes"].keys():
                        tracker_info[obj_id]["classes"][cls] = score
                    else:
                        tracker_info[obj_id]["classes"][cls] += score
                    tracker_info[obj_id]["timestamp"] = timestamp
                    totalscore = sum(tracker_info[obj_id]["classes"].values())
                    old_score = totalscore - score
                    # Update bb coordinates
                    old_weighted_coords = np.asarray(tracker_info[obj_id]["bbox"]) * old_score
                    new_weighted_coords = np.asarray(bbox) * score
                    tracker_info[obj_id]["bbox"] = ((old_weighted_coords + new_weighted_coords) / totalscore).tolist()
                    # tracker_info[obj_id]["bbox"] = bbox
                    tracker_info[obj_id]["bbox_xyxy"] = DlibTracker.xywh_box_to_xyxy_box(tracker_info[obj_id]["bbox"])
                    # Update class score
                    updated_trackers.append(obj_id)
            if is_new_object:
                obj_id = self.tracker_count
                self.tracker_count += 1
                bbox_xyxy = DlibTracker.xywh_box_to_xyxy_box(bbox)
                self.tracker_info[obj_id] = {"bbox": bbox, "score": score, "cls": cls,
                                            "timestamp": timestamp, "classes": {cls: score},
                                            "bbox_xyxy": bbox_xyxy}
                updated_trackers.append(obj_id)

        # Update trackers. Update new bbox, class and score.
        for img_timestamp in sorted(self.img_stream_queue.keys()):
            img = self.img_stream_queue[img_timestamp]
            if img_timestamp < bb_timestamp:
                continue

            for object_id, t in self.tracker_info.items():

                if object_id not in updated_trackers:
                    continue
                self.trackers[object_id] = dlib.correlation_tracker()
                drbbox = tracker_info[object_id]["bbox_xyxy"]
                drectangle = dlib.rectangle(int(drbbox[0]), int(drbbox[1]), int(drbbox[2]), int(drbbox[3]))
                self.trackers[object_id].start_track(img, drectangle)
                self.trackers[object_id].update(img)
                bb = self.trackers[object_id].get_position()
                bbox = DlibTracker.xyxy_box_to_xywh_box([bb.left(), bb.top(), bb.right(), bb.bottom()])
                self.tracker_info[object_id]["bb"] = bbox
                self.tracker_info[object_id]["bb_xyxy"] = DlibTracker.xywh_box_to_xyxy_box(bbox)
                # Apply score decay, so that new matched detections have a higher impact on the position of the bbox.
                for cls in self.tracker_info[object_id]["classes"]:
                    self.tracker_info[object_id]["classes"][cls] /= self.total_score_decay
            self.tracker_info_history[img_timestamp] = self.tracker_info

        self.tracker_alignment_running = False


    def update_trackers(self, img, timestamp):
        while self.tracker_alignment_running == True:
            time.sleep(0.001)
        self.tracker_update_running = True

        # First, delete all trackers that have a low total score.
        for object_id, t_info in self.tracker_info.items():
            totalscore = sum(self.tracker_info[object_id]["classes"].values())
            if totalscore < self.total_score_threshold:
                print("Tracker for object {} has a low score of {}. It will be removed.".format(str(object_id), str(totalscore)))
                del self.trackers[object_id]
                del self.tracker_info[object_id]

        # Now update current bbs
        self.current_bbs = {}
        for object_id, t in self.trackers.items():
            self.trackers[object_id].update(img)
            bb = t.get_position()
            bbox = self.xyxy_box_to_xywh_box([bb.left(), bb.top(), bb.right(), bb.bottom()])
            classes = self.tracker_info[object_id]["classes"]
            cls = max(classes, key=classes.get)
            score = classes[cls]
            self.tracker_info[object_id]["bb"] = bbox
            self.tracker_info[object_id]["bb_xyxy"] = bb
            # Apply totalscore decay, so that new matched detections have a higher impact on the position of the bbox in align_detections_and_trackers function.
            for cls in self.tracker_info[object_id]["classes"]:
                self.tracker_info[object_id]["classes"][cls] /= self.total_score_decay
            if object_id not in self.current_bbs.keys():
                self.current_bbs[object_id] = {}
                self.current_bbs[object_id]["label"] = object_id
                self.current_bbs[object_id]["bbox"] = bbox
                self.current_bbs[object_id]["score"] = score
                self.current_bbs[object_id]["timestamp"] = timestamp
                self.current_bbs[object_id]["class"] = cls
                self.current_bbs[object_id]["classes"] = classes
        self.tracker_info_history[timestamp] = self.tracker_info
        self.tracker_update_running = False

    def __init__(self):
        self.trackers = {}
        self.tracker_info = {}
        self.tracker_count = 0
        self.bbs_received = 0
        self.total_score_threshold = 0.5
        self.total_score_decay = 1.1
        self.iou_threshold = 0.5
        self.tracker_alignment_running = False
        self.tracker_update_running = False
        Tracker.__init__(self)

