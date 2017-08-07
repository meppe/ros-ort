#!/usr/bin/env python
import cv2
import os
import _init_paths
from lib.datasets.factory import get_imdb
import lib.datasets.imdb
import argparse
import pprint
import numpy as np
import sys
from lib.datasets.nico import Nico
from frcnn.annotator import Annotator
from lib.annotation.generate_train_val_test import generate_train_val_test_split

'''
Data generation
    Iterate through videos
        For each frame in video
            Update dlib trackers
            Let the user draw bounding boxes for objects that are not yet in the image
            Let the user delete bounding boxes that are not appropriate any more
            Save image with all bounding boxes
'''

DATA_ROOT = "/storage/data/nico2017"

if __name__ == '__main__':
    print("Starting annotator. The tool reads single frames. To generate frames from video, you can use the gen_frames.sh script")
    annotator = Annotator(data_root=DATA_ROOT, classes=Nico.CLASSES)
    generate_train_val_test_split()


