# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi he, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from datasets.bib_detect import bib_detect
from model.detect import detect, _remove_overlapping_boxes
from model.config import cfg, cfg_from_file, cfg_from_list
import argparse
import pprint
import time, os, sys

import tensorflow as tf
from nets.vgg16 import vgg16
from nets.res101 import Resnet101
from nets.numrecog import BibRecogNetwork

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Streamlining the \
            dection: 1. detect bib area using a Fast R-CNN network; \
            2. recognize the bib using a BibCog network')
  parser.add_argument('--cfg', dest='cfg_file',
            help='optional config file', default=None, type=str)
  parser.add_argument('--model', dest='model',
            help='model(s) to test. If provided one model then we \
            only do Faster R-CNN. If two models are provided then \
            we do both Faster R-CNN and BibRecognition',
            type=str, nargs='+')
  parser.add_argument('--source_files', dest='source_files',
            help='file or folder of the to be detected image(s)',
            type=str)
  parser.add_argument('--output_fn', dest='output_fn',
            help='output file path',
            default=None, type=str)
  parser.add_argument('--num_dets', dest='max_per_image',
            help='max number of detections per image',
            default=100, type=int)
  parser.add_argument('--tag', dest='tag',
                        help='tag of the model',
                        default='', type=str)
  parser.add_argument('--faster_rcnn_net', dest='faster_rcnn_net',
                      help='vgg16 or res101',
                      default='res101', type=str)

  parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args

if __name__ == '__main__':
    boxes = [[10, 10, 50, 70], [8, 10, 30, 20], [90, 90, 99, 99]]
    print(_remove_overlapping_boxes(boxes))

    exit()

    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
      cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
      cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    tag = args.tag
    tag = tag if tag else 'default'

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if args.faster_rcnn_net == 'vgg16':
      faster_rcnn_net = vgg16(batch_size=1)
      faster_rcnn_prefix = 'vgg_16'
    elif args.faster_rcnn_net == 'res101':
      faster_rcnn_net = Resnet101(batch_size=1)
      faster_rcnn_prefix = 'resnet_v1_101'
    else:
      raise NotImplementedError
    # load model

    faster_rcnn_net.create_architecture(sess, "TEST", 2,  
                            tag='default', anchor_scales=cfg.ANCHOR_SCALES)
    faster_rcnn_vars = [v for v in tf.global_variables() 
                          if v.name.startswith(faster_rcnn_prefix)]
    faster_rcnn_saver = tf.train.Saver(faster_rcnn_vars)
    faster_rcnn_saver.restore(sess, args.model[0])


    num_recog_net = BibRecogNetwork(args.max_per_image)
    num_recog_net.build_network()
    num_recog_vars = [v for v in tf.global_variables() 
                          if not v.name.startswith(faster_rcnn_prefix)]
    num_recog_saver = tf.train.Saver(num_recog_vars)
    num_recog_saver.restore(sess, args.model[1])

    print('Models Loaded.')

    imdb = bib_detect(args.source_files)
    detect(sess, faster_rcnn_net, imdb, 
        output_fn=args.output_fn,
        num_recog_net=num_recog_net, 
        max_per_image=args.max_per_image)

    sess.close()
