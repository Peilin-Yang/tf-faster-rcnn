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
from model.config import cfg, cfg_from_file, cfg_from_list
import argparse
import pprint
import time, os, sys

import tensorflow as tf
from nets.vgg16 import vgg16
from nets.res101 import Resnet101

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
            required=True, type=str, nargs='+')
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
    elif args.faster_rcnn_net == 'res101':
      faster_rcnn_net = Resnet101(batch_size=1)
    else:
      raise NotImplementedError
    # load model

    faster_rcnn_net.create_architecture(sess, "TEST", 2,  
                            tag='default', anchor_scales=cfg.ANCHOR_SCALES)

    print(('Loading model check point from {:s}').format(args.model))
    print([type(v.name) for v in tf.global_variables()])
    print([v.name for v in tf.global_variables()])
    print(args.faster_rcnn_net)
    print(len(args.model), args.model)
    faster_rcnn_vars = [v.name for v in tf.global_variables() 
                          if v.name.startswith(unicode(args.faster_rcnn_net))]
    print(faster_rcnn_vars)
    faster_rcnn_saver = tf.train.Saver(faster_rcnn_vars)
    faster_rcnn_saver.restore(sess, args.model[0])
    print('Loaded.')

    # test_net(sess, faster_rcnn_net, imdb, filename, max_per_image=args.max_per_image)

    sess.close()
