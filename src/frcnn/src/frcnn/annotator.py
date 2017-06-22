import time
import errno
import sys
import scipy
import os
import numpy as np
import time
import dlib
import cv2
from shutil import copyfile
# import Image
# from cv_bridge import CvBridge
import lib.annotation.selectinwindow as selectinwindow

class Annotator:

    def __init__(self, data_root, classes=[]):
        self.data_root = data_root
        self.object_classes = list(classes)
        sys.setrecursionlimit(10 ** 9)
        self.data_path = self.data_root + '/generate_frames/video_frames'
        self.image_path = self.data_root + '/nico2017/JPEGImages'
        self.annotation_path = self.data_root + '/nico2017/Annotations'
        self.trackers = []
        self.sleep_time = 0.1
        self.next_tracker_id = 0
        self.current_sample_id = self.get_next_sample_id()
        self.stop_annotate = False
        for subdir, dirs, _ in os.walk(self.data_path):
            for dir in dirs:
                self.trackers = []
                self.next_tracker_id = 0
                for _, _, files in os.walk(self.data_path+os.sep+dir):
                    for file in sorted(files):
                        filepath = subdir +os.sep + dir + os.sep + file
                        if filepath.endswith(".jpg"):
                            self.update_bbs(filepath)
                        self.write_data(filepath)
                        if self.stop_annotate == True:
                            exit(1)

    def get_next_sample_id(self):
        next_id = 0
        for subdir, dirs, files in os.walk(self.annotation_path):
            for file_name in files:
                number = int(file_name.split(".")[0])
                next_id = max(next_id, number)
        return next_id

    def write_data(self, filepath):
        if len(self.trackers) == 0:
            return
        self.current_sample_id += 1
        img_filename = str(self.current_sample_id).zfill(6)+'.jpg'
        img_path = self.image_path+os.sep+img_filename
        xml_filename = str(self.current_sample_id).zfill(6)+'.xml'
        xml_path = self.annotation_path + os.sep + xml_filename
        img = cv2.imread(filepath, 1)
        (img_height, img_width, img_depth) = img.shape
        xml_str = "<annotation>\n" + \
                  "\t<folder>{}</folder>\n".format("nico2017") + \
                  "\t<filename>{}</filename>".format(img_filename) + \
                  "\t<source>{}</source>\n".format(filepath) + \
                  "\t<owner>Manfred Eppe, Knowledge Technology Group, University of Hamburg</owner>\n" + \
                  "\t<size>\n" + \
                  "\t\t<width>{}</width>\n".format(img_width) + \
                  "\t\t<height>{}</height>\n".format(img_height) + \
                  "\t\t<depth>{}</depth>\n".format(img_depth) + \
                  "\t</size>\n" + \
                  "\t<segmented>0</segmented>\n"
        for tracker in self.trackers:
            object_classes = tracker["object_classes"]
            bbox = tracker["bbox"]
            for object_class in object_classes:
                xml_str += "\t<object>\n" + \
                           "\t\t<name>{}</name>\n".format(object_class) + \
                           "\t\t<pose></pose>\n" + \
                           "\t\t<truncated>1</truncated>\n" + \
                           "\t\t<difficult>0</difficult>\n" + \
                           "\t\t<bndbox>\n" + \
                           "\t\t\t<xmin>{}</xmin>\n".format(bbox[0]) + \
                           "\t\t\t<ymin>{}</ymin>\n".format(bbox[1]) + \
                           "\t\t\t<xmax>{}</xmax>\n".format(bbox[2]) + \
                           "\t\t\t<ymax>{}</ymax>\n".format(bbox[3]) + \
                           "\t\t</bndbox>\n" + \
                           "\t</object>\n"
        xml_str += "</annotation>\n"

        xml_file = open(xml_path, "w")
        xml_file.write(xml_str)
        copyfile(filepath, img_path)


    def update_bbs(self, filepath):
        print("updating BBs for frame {}".format(filepath))
        key = None
        img = cv2.imread(filepath, 1)
        (img_height, img_width, img_depth) = img.shape
        time.sleep(1)
        for i,tracker in enumerate(self.trackers):
            self.trackers[i]["tracker"].update(img)
            drect = self.trackers[i]["tracker"].get_position()
            bbox = [max(0,int(drect.left())), max(0,int(drect.top())), min(int(drect.right()),img_width), min(int(drect.bottom()),img_height)]

            self.trackers[i]["bbox"] = bbox
        while key != "n":
            print(
            "While window is active, press <a> to add a new bounding box (a new window will open), " +
            "<d> to delete a bounding box, <n> for next frame and <q> to stop annotation.")
            # print("Opening display window...")
            img = cv2.imread(filepath, 1)
            time.sleep( self.sleep_time)
            key_int = self.show_img_with_bbs(img)
            if key_int in range(256):
                key = chr(key_int)
            else:
                print("Invalid input, please select again")
                continue
            if key == "a":
                time.sleep(self.sleep_time)
                # print("Destroying all windows...")
                cv2.destroyAllWindows()
                time.sleep(self.sleep_time)
                # print("Opening drawing window...")
                self.add_bb(filepath, img)
            elif key == "d":
                print("Type number of bounding box to delete in console and press ENTER:")
                bb_num = raw_input()
                print ("you entered {}".format(bb_num))
                for idx, tracker in enumerate(self.trackers):
                    if str(tracker["id"]) == bb_num:
                        print("Deleting tracker for object {}".format(bb_num))
                        del self.trackers[idx]
                        time.sleep( self.sleep_time)
                        # print("Destroying all windows...")
                        cv2.destroyAllWindows()
                        time.sleep( self.sleep_time)
                        break
            elif key == "q":
                self.stop_annotate = True
                break
            time.sleep(self.sleep_time)
            print("destroying all windows")
            cv2.destroyAllWindows()
            time.sleep(self.sleep_time)

    def show_img_with_bbs(self, img):
        print("The current trackers are {}".format(self.trackers))
        for tracker in self.trackers:
            bbox = tracker["bbox"]
            ul = (bbox[0], bbox[1])
            lr = (bbox[2], bbox[3])
            rect_color = (0, 0, 255)
            font_color = (255, 255, 255)
            cv2.rectangle(img, ul, lr, rect_color, 2)
            bbox_text = []
            bbox_text.append("id: {}, classes {}".format(tracker["id"], tracker["object_classes"]))
            font_height = 12
            txt_ul = [ul[0], ul[1] - (len(bbox_text) * font_height)]
            txt_ul[0] = max(txt_ul[0], 0)
            txt_ul[1] = max(txt_ul[1], font_height)
            for i, txt in enumerate(bbox_text):
                line_ul = (txt_ul[0], txt_ul[1] + font_height * i)
                cv2.putText(img, txt, line_ul, cv2.FONT_HERSHEY_SIMPLEX, 0.4, font_color, 1)
        cv2.imshow("image", img)
        key = cv2.waitKey(0)
        return key

    def add_bb(self, filepath, img):
        selected = False
        while selected == False:
            print("Draw a rectangle and double click to finish selection.")
            x, y, w, h = self.draw_box(filepath)
            bbox = [x, y, x+w, y+h]
            print("You drew a bounding box with the coordinates [left, top, right, bottom]={}." + \
                  "\nNow please enter one or more object classes for the bounding box, separated by comma, and without whitespaces.".format(bbox))
            if len(self.object_classes) > 0:
                print("Possible object classes are: {}".format(self.object_classes))
            object_classes = raw_input().replace(" ", "").split(",")
            if len(self.object_classes) == 0:
                selected = True
            else:
                selected = True
                for o in object_classes:
                    if o not in self.object_classes:
                        print("Error, object {} not in valid classes. Please repeat annotation!".format(o))
                        selected = False

        new_tracker = {}
        new_tracker["id"] = self.next_tracker_id
        self.next_tracker_id += 1
        new_tracker["object_classes"] = object_classes
        new_tracker["tracker"] = dlib.correlation_tracker()
        new_tracker["bbox"] = bbox
        drectangle = dlib.rectangle(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        new_tracker["tracker"].start_track(img, drectangle)
        self.trackers.append(new_tracker)

    def draw_box(self, filepath):
        # print("1")
        rectI = selectinwindow.dragRect
        # print("file ", filepath)
        img = cv2.imread(filepath)
        # print("image read")
        wName = 'Select new bb for {}'.format(filepath)
        im_height = img.shape[0]
        im_width = img.shape[1]
        selectinwindow.init(rectI, img, wName, im_width, im_height)
        # print("selected")
        cv2.namedWindow(rectI.wname)
        # print("2")
        cv2.setMouseCallback(rectI.wname, selectinwindow.dragrect, rectI)
        # print("3")
        # print("Opening drawing window...")
        # time.sleep(0.5)
        while True:
            # display the image
            cv2.imshow(wName, rectI.image)
            key = cv2.waitKey(1) & 0xFF
            # print("key: {}".format(key))
            # if returnflag is True, break from the loop
            if rectI.returnflag == True:
                # print("returnflag")
                break
            # print("4")
        # rectI.returnflag == False

        # print "Dragged rectangle coordinates"
        print str(rectI.outRect.x) + ',' + str(rectI.outRect.y) + ',' + \
              str(rectI.outRect.w) + ',' + str(rectI.outRect.h)

        # close all open windows
        cv2.destroyAllWindows()
        return rectI.outRect.x, rectI.outRect.y, rectI.outRect.w, rectI.outRect.h






