#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths
from lib.fast_rcnn.train import get_training_roidb, train_net
from lib.fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from lib.datasets.factory import get_imdb
import lib.datasets.imdb
import caffe
import pprint
import numpy as np
import sys
import argparse

BASE_DIR = "/opt/ros-ort/src/frcnn/src/"

def parse_args():
    ###################
    # Default values: #
    ###################

    # net = "VGG16"
    net = "ZF"

    dataset = "nico"
    # train_imdb = "nico_2017_trainval"
    train_imdb = "nico_2017_all"

    # dataset = "pascal_voc"
    # train_imdb = "voc_2007_trainval"

    pt_dir = dataset
    iters = 70000
    cfg = 'experiments/cfgs/faster_rcnn_end2end.yml'

    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=BASE_DIR+'models/'+pt_dir+'/'+net+'/faster_rcnn_end2end/solver.prototxt', type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=iters, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=BASE_DIR+'/data/imagenet_models/'+net+'.v2.caffemodel', type=str)
    # parser.add_argument('--weights', dest='pretrained_model',
    #                     help='initialize with pretrained model weights',
    #                     default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=BASE_DIR+'/'+cfg, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default=train_imdb, type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()

    args = parser.parse_args()
    return args

def combined_roidb(imdb_names):
    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print 'Loaded dataset `{:s}` for training'.format(imdb.name)
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        imdb = datasets.imdb.imdb(imdb_names)
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)


    # set up caffe
    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)

    imdb, roidb = combined_roidb(args.imdb_name)
    print '{:d} roidb entries'.format(len(roidb))

    output_dir = get_output_dir(imdb)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    train_net(args.solver, roidb, output_dir,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters)
