# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Set up paths for Fast R-CNN."""

import os.path as osp
import sys
import subprocess

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add caffe to PYTHONPATH
caffe_path = osp.join(this_dir, '..', 'src', 'py-faster-rcnn', 'caffe-fast-rcnn', 'python')
add_path(caffe_path)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, '..', 'src', 'lib')
add_path(lib_path)

# Add src to PYTHONPATH
src_path = osp.join(this_dir, '..', 'src')
add_path(src_path)

# Add ros to PYTHONPATH
ros_path = "/opt/ros/kinetic/lib/python2.7/dist-packages"
add_path(ros_path)

# Add ros-ort to PYTHONPATH
ros_path = "/opt/ros-ort/devel/lib/python2.7/dist-packages"
add_path(ros_path)

sys.path.append("/opt/ros-ort/install/lib/python2.7/dist-packages")
sys.path.append("/opt/ros-ort/src/frcnn/src")
sys.path.append("/opt/ros/kinetic/bin")
#
# subprocess.call(['/frcnn_entrypoint.sh'])