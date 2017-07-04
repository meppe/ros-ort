from frcnn.tracker import Tracker
import numpy as np
import time
import dlib
from threading import Lock
import copy

class DlibTracker(Tracker):

    # @staticmethod
    # def xywh_box_to_xyxy_box(bb):
    #     """Converts from center_x, center_y, width, height to ul_x, ul_y, lr_x, lr_y"""
    #     """(ul_x, ul_y) are (0,0) in the top left corner"""
    #     x = bb[0]
    #     y = bb[1]
    #     w = bb[2]
    #     h = bb[3]
    #     return np.array([int(x-(w/2.)), int(y-(h/2.)), int(x+(w/2.)), int(y+(h/2.))]).tolist()
    #
    # @staticmethod
    # def xyxy_box_to_xywh_box(bb):
    #     """Converts from center_x, center_y, width, height to ul_x, ul_y, lr_x, lr_y"""
    #     """(ul_x, ul_y) are (0,0) in the top left corner"""
    #     x1 = bb[0]
    #     y1 = bb[1]
    #     x2 = bb[2]
    #     y2 = bb[3]
    #     return np.array([int(x1 + ((x2-x1) / 2)), int(y1 + ((y2-y1) / 2)), int((x2-x1)), int((y2-y1))]).tolist()

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

    # @profile
    def align_detections_and_trackers(self, bbs):
        '''
        This is called when a new bounding box has been received.
        :param bbs:
        :return:
        '''

        bb_timestamp = self.last_detected_bb_timestamp

        print("Aligning trackers with new BB for timestamp {}".format(bb_timestamp))

        while len(self.img_stream_queue.keys()) == 0:
            print("Waiting for img queue to fill.")
            time.sleep(0.01)

        assert(len(self.img_stream_queue.keys()) != 0), "Error! Empty image queue!"

        closest_timestamp = min(self.img_stream_queue.keys(), key=lambda x: abs(x-bb_timestamp))

        if closest_timestamp != bb_timestamp:
            print("Frame {} has not been received. Processing the closest frame {} instead.".format(
                bb_timestamp, closest_timestamp))

        updated_trackers = []
        for label, bb in bbs.items():
            bbox = bb["bbox"]
            score = bb["score"]
            cls = bb["class"]

            is_new_object = True
            # Iterate through all tracker states at the timestamp of the BB and compute iou.
            # If IOU is below threshold, match it.
            ious = {}
            for obj_id, t_info in self.tracker_info.items():
                ious[obj_id] = DlibTracker.iou(t_info["bbox"], bbox)
            # Get obj_id that has the closest bb:
            if len(ious.keys()) > 0:
                obj_id = max(ious, key=ious.get)
                max_iou = ious[obj_id]
                if max_iou > self.iou_threshold:
                    is_new_object = False
                    if cls not in self.tracker_info[obj_id]["classes"].keys():
                        # Add a new class
                        self.tracker_info[obj_id]["classes"][cls] = score
                    else:
                        # Add score to existing class
                        self.tracker_info[obj_id]["classes"][cls] += score
                    self.tracker_info[obj_id]["timestamp"] = bb_timestamp
                    totalscore = sum(self.tracker_info[obj_id]["classes"].values())
                    old_score = totalscore - score
                    # Update bb coordinates
                    old_weighted_coords = np.asarray(self.tracker_info[obj_id]["bbox"]) * old_score
                    new_weighted_coords = np.asarray(bbox) * score
                    self.tracker_info[obj_id]["bbox"] = ((old_weighted_coords + new_weighted_coords) / totalscore).tolist()
                    # self.tracker_info[obj_id]["bbox"] = bbox
                    self.tracker_info[obj_id]["bbox"] = [int(round(coord,0)) for coord in self.tracker_info[obj_id]["bbox"]]
                    updated_trackers.append(obj_id)
            if is_new_object:
                obj_id = self.tracker_count
                self.tracker_count += 1
                self.tracker_info[obj_id] = {"bbox": bbox, "score": score, "cls": cls,
                                            "timestamp": bb_timestamp, "classes": {cls: score}}
                updated_trackers.append(obj_id)

        # Restart those trackers that have been realigned with new detections and track the frames between BB detection
        # and last frame.
        for object_id, t in self.tracker_info.items():
            if object_id not in updated_trackers:
                continue
            # Restart tracker
            self.trackers[object_id] = dlib.correlation_tracker()
            drbbox = self.tracker_info[object_id]["bbox"]
            drectangle = dlib.rectangle(int(drbbox[0]), int(drbbox[1]), int(drbbox[2]), int(drbbox[3]))
            img = self.img_stream_queue[closest_timestamp]
            tracker = self.trackers[object_id]
            tracker.start_track(img, drectangle)
            # Re-track the frames between the bb-detection frame and the last received frame.
            last_img_timestamp = bb_timestamp
            for img_timestamp in sorted(self.img_stream_queue.keys()):
                if img_timestamp <= bb_timestamp:
                    continue
                img = self.img_stream_queue[img_timestamp]
                self.trackers[object_id].update(img)
                last_img_timestamp = img_timestamp
            # Update tracker info
            bbox = self.trackers[object_id].get_position()
            self.tracker_info[object_id]["bbox"] = [int(bbox.left()), int(bbox.top()), int(bbox.right()),
                                                    int(bbox.bottom())]
            self.tracker_info[object_id]["timestamp"] = last_img_timestamp
            # Apply score decay, so that new matched detections have a higher impact on the position of the bbox.
            # print("Start iteration through tracker_info in align_trackers")
            # print self.tracker_info[object_id]["classes"]

            self.tracker_info_lock.acquire()
            try:
                new_tracker_info = copy.deepcopy(self.tracker_info[object_id])
            finally:
                self.tracker_info_lock.release()

            for cls in new_tracker_info["classes"]:
                new_tracker_info["classes"][cls] /= self.total_score_decay

            self.tracker_info_lock.acquire()
            try:
                self.tracker_info[object_id] = new_tracker_info
            finally:
                self.tracker_info_lock.release()

        self.delete_low_score_trackers()

    def delete_low_score_trackers(self):
        # Mark all trackers that have a low total score for deletion.
        tracker_scores = {}
        trackers_to_delete = set()
        num_trackers = len(self.tracker_info.keys())
        self.tracker_info_lock.acquire()
        self.trackers_lock.acquire()
        try:
            for object_id in self.tracker_info.keys():
                totalscore = sum(self.tracker_info[object_id]["classes"].values())
                tracker_scores[object_id] = totalscore
                # TODO: Trackers should only be removed if there are too many, so this is commented out for now.
                # cls_to_del = []
                # for cls in self.tracker_info[object_id]["classes"]:
                #     if self.tracker_info[object_id]["classes"][cls] < self.class_threshold:
                #         cls_to_del.append(cls)
                # for cls in cls_to_del:
                #     del self.tracker_info[object_id]["classes"][cls]
                # if totalscore < self.cum_threshold:
                #     print("Tracker for object {} has a low score of {}. It will be removed.".format(
                #         str(object_id), str(totalscore)))
                #     trackers_to_delete.add(object_id)
                # else:
                #     remaining_tracker_scores[object_id] = totalscore

            # If there are too many trackers, mark those with the lowest score for deletion.
            while num_trackers - len(trackers_to_delete) > self.max_trackers:
                t_to_del = min(tracker_scores, key=tracker_scores.get)
                print("Only {} trackers allowed, deleting tracker for object {}."
                      .format(str(self.max_trackers), str(t_to_del)))
                trackers_to_delete.add(t_to_del)
                del tracker_scores[t_to_del]

            # Now delete all trackers to delete
            for object_id in trackers_to_delete:
                assert(object_id in self.trackers.keys()), "Object id not in trackers list"
                del self.trackers[object_id]
                assert(object_id in self.tracker_info.keys()), "Object id not in tracker_info list"
                del self.tracker_info[object_id]
        finally:
            self.tracker_info_lock.release()
            self.trackers_lock.release()

    # @profile
    def update_trackers(self, img, timestamp):
        '''
        This is called when a new frame has been received.
        :param img:
        :param timestamp:
        :return:
        '''

        print("Updating trackers with new image for timestamp {}".format(timestamp))

        self.current_bbs = {}
        self.trackers_lock.acquire()
        try:
            for object_id in self.trackers.keys():
                self.trackers[object_id].update(img)
                bb = self.trackers[object_id].get_position()
                bbox = [bb.left(), bb.top(), bb.right(), bb.bottom()]
                bbox = [int(coord) for coord in bbox]
                classes = self.tracker_info[object_id]["classes"]
                self.tracker_info[object_id]["bb"] = bbox
                # Apply totalscore decay, so that new matched detections have a higher impact on the position of the bbox
                # in align_detections_and_trackers function.
                # self.tracker_info_lock.acquire()
                # try:
                for cls in self.tracker_info[object_id]["classes"]:
                    self.tracker_info[object_id]["classes"][cls] /= self.total_score_decay
                # finally:
                #     self.tracker_info_lock.release()
                score = sum(classes.values())
                # Update current_bbs. This is the dict that is being visualized through parent class Tracker.
                # if object_id not in self.current_bbs.keys():
                self.current_bbs[object_id] = {}
                self.current_bbs[object_id]["label"] = object_id
                self.current_bbs[object_id]["bbox"] = bbox
                self.current_bbs[object_id]["score"] = score
                self.current_bbs[object_id]["timestamp"] = timestamp
                self.current_bbs[object_id]["classes"] = classes
        finally:
            self.trackers_lock.release()
            # self.tracker_info_lock.release()

    def __init__(self, args):
        self.tracker_info_lock = Lock()
        self.trackers_lock = Lock()
        self.trackers = {}
        self.tracker_info = {}
        self.tracker_count = 0
        self.bbs_received = 0
        self.cum_threshold = args.cum_threshold
        self.class_threshold = args.class_threshold
        self.total_score_decay = 1.1
        # Distance threshold to start a new tracker. The higher the number the more bounding boxes there are.
        self.iou_threshold = 0.2
        self.max_trackers = args.max_trackers
        self.tracker_alignment_running = False
        self.tracker_update_running = False
        print ("Running tracker with arguments {}.".format(args))
        Tracker.__init__(self, mask_objects=args.mask_objects, write_to_file=args.write_to_file, classes_to_display=[])

