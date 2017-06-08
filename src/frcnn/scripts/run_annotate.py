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


'''
Data generation
    Iterate through videos
        For each frame in video
            Update dlib trackers
            Let the user draw bounding boxes for objects that are not yet in the image
            Let the user delete bounding boxes that are not appropriate any more
            Save image with all bounding boxes
'''

def generate_ann_files():
    pass

if __name__ == '__main__':
    annotator = Annotator(Nico.CLASSES)
    generate_ann_files()


