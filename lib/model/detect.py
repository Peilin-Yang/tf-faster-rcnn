# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
try:
  import cPickle as pickle
except ImportError:
  import pickle
import os
import math

from utils.timer import Timer
from utils.cython_nms import nms, nms_new
from utils.boxes_grid import get_boxes_grid
from utils.blob import im_list_to_blob

from model.config import cfg, get_output_dir
from model.bbox_transform import clip_boxes, bbox_transform_inv

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

def _get_blobs(im):
  """Convert an image and RoIs within that image into network inputs."""
  blobs = {}
  blobs['data'], im_scale_factors = _get_image_blob(im)

  return blobs, im_scale_factors

def _clip_boxes(boxes, im_shape):
  """Clip boxes to image boundaries."""
  # x1 >= 0
  boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
  return boxes

def _rescale_boxes(boxes, inds, scales):
  """Rescale boxes according to image rescaling."""
  for i in range(boxes.shape[0]):
    boxes[i,:] = boxes[i,:] / scales[int(inds[i])]

  return boxes

def _non_max_suppression_fast(boxes, overlapThresh):
  """
  merge the overlapping rectangles
  """
  # if there are no boxes, return an empty list
  if len(boxes) == 0:
    return []

  # if the bounding boxes integers, convert them to floats --
  # this is important since we'll be doing a bunch of divisions
  if boxes.dtype.kind == "i":
    boxes = boxes.astype("float")
  #  
  # initialize the list of picked indexes   
  pick = []

  # grab the coordinates of the bounding boxes
  x1 = boxes[:,0]
  y1 = boxes[:,1]
  x2 = boxes[:,2]
  y2 = boxes[:,3]

  # compute the area of the bounding boxes and sort the bounding
  # boxes by the bottom-right y-coordinate of the bounding box
  area = (x2 - x1 + 1) * (y2 - y1 + 1)
  idxs = np.argsort(y2)

  # keep looping while some indexes still remain in the indexes
  # list
  while len(idxs) > 0:
    # grab the last index in the indexes list and add the
    # index value to the list of picked indexes
    last = len(idxs) - 1
    i = idxs[last]
    pick.append(i)

    # find the largest (x, y) coordinates for the start of
    # the bounding box and the smallest (x, y) coordinates
    # for the end of the bounding box
    xx1 = np.maximum(x1[i], x1[idxs[:last]])
    yy1 = np.maximum(y1[i], y1[idxs[:last]])
    xx2 = np.minimum(x2[i], x2[idxs[:last]])
    yy2 = np.minimum(y2[i], y2[idxs[:last]])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    # compute the ratio of overlap
    overlap = (w * h) / area[idxs[:last]]

    # delete all indexes from the index list that have
    idxs = np.delete(idxs, np.concatenate(([last],
       np.where(overlap > overlapThresh)[0])))
    print(idxs)

  # return only the bounding boxes that were picked using the
  # integer data type
  return boxes[pick]

def im_detect(sess, net, im):
  blobs, im_scales = _get_blobs(im)
  assert len(im_scales) == 1, "Only single-image batch implemented"

  im_blob = blobs['data']
  # seems to have height, width, and image scales
  # still not sure about the scale, maybe full image it is 1.
  blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

  _, scores, bbox_pred, rois = net.test_image(sess, blobs['data'], blobs['im_info'])
  
  boxes = rois[:, 1:5] / im_scales[0]
  # print(scores.shape, bbox_pred.shape, rois.shape, boxes.shape)
  scores = np.reshape(scores, [scores.shape[0], -1])
  bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
  if cfg.TEST.BBOX_REG:
    # Apply bounding-box regression deltas
    box_deltas = bbox_pred
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = _clip_boxes(pred_boxes, im.shape)
  else:
    # Simply repeat the boxes, once for each class
    pred_boxes = np.tile(boxes, (1, scores.shape[1]))

  return scores, pred_boxes

def apply_nms(all_boxes, thresh):
  """Apply non-maximum suppression to all predicted boxes output by the
  test_net method.
  """
  num_classes = len(all_boxes)
  num_images = len(all_boxes[0])
  nms_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
  for cls_ind in range(num_classes):
    for im_ind in range(num_images):
      dets = all_boxes[cls_ind][im_ind]
      if dets == []:
        continue

      x1 = dets[:, 0]
      y1 = dets[:, 1]
      x2 = dets[:, 2]
      y2 = dets[:, 3]
      scores = dets[:, 4]
      inds = np.where((x2 > x1) & (y2 > y1) & (scores > cfg.TEST.DET_THRESHOLD))[0]
      dets = dets[inds,:]
      if dets == []:
        continue

      keep = nms(dets, thresh)
      if len(keep) == 0:
        continue
      nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
  return nms_boxes


def crop_blobs(im, detections, img_w=64, img_h=64, channels=3, padding=5):
    blobs = np.zeros((detections.shape[0], img_w, img_h, channels),
                  dtype=np.float32)
    for k in xrange(detections.shape[0]):
        bbox = detections[k]
        crop_img = im[max(1, int(bbox[1]-padding)):min(im.shape[0], int(bbox[3]+padding)), 
                    max(1, int(bbox[0]-padding)):min(im.shape[1], int(bbox[2]+padding))]
        # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
        resized = cv2.resize(crop_img, (img_w, img_h))
        blobs[k, 0:img_w, 0:img_h, :] = resized

    return blobs

def num_recognition(sess, num_recog_net, blobs):
    all_recognitions = []

    _t = Timer()
    _t.tic()
    res = num_recog_net.predict(sess, blobs)
    _t.toc()

    for i in range(blobs.shape[0]):
        l = res[i].tolist()
        num = ''.join([str(n) for n in l[1:l[0]+2]])
        all_recognitions.append('%s' % (num))

    return all_recognitions

def detect(sess, faster_rcnn_net, imdb, num_recog_net=None, 
      max_per_image=100, thresh=0.05, output_fn=None):
    np.random.seed(cfg.RNG_SEED)
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #  all_boxes[cls][image] = N x 5 array of detections in
    #  (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
         for _ in range(imdb.num_classes)]
    num_recognitions = [[[] for _ in range(num_images)]
         for _ in range(imdb.num_classes)]

    # timers
    _t = {'im_detect' : Timer(), 'recog': Timer(), 'misc' : Timer()}

    for i in range(num_images):
        im = cv2.imread(imdb.image_path_at(i))

        _t['im_detect'].tic()
        scores, boxes = im_detect(sess, faster_rcnn_net, im)
        _t['im_detect'].toc()

        _t['misc'].tic()

        # skip j = 0, because it's the background class
        for j in range(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
            .astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            all_boxes[j][i] = cls_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                        for j in range(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        # merge the overlapping boxes
        for j in range(1, imdb.num_classes):
          non_overlapping_boxes = _non_max_suppression_fast(all_boxes[j][i][:,:4], 0)
          all_boxes[j][i][:,:4] = non_overlapping_boxes


        _t['misc'].toc()

        print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
            .format(i + 1, num_images, _t['im_detect'].average_time,
              _t['misc'].average_time))

        # number recognition
        if num_recog_net:
            _t['recog'].tic()
            for j in range(1, imdb.num_classes):
                dets = all_boxes[j][i]
                if dets == []:
                    continue 
                cropped_blobs = crop_blobs(im, dets)
                recognitions = num_recognition(sess, num_recog_net, cropped_blobs)
                for k in xrange(all_boxes[j][i].shape[0]):
                    num_recognitions[j][i].append(recognitions[k])
            _t['recog'].toc()

    if output_fn:
        of = open(output_fn, 'w')

    for cls_ind, cls in enumerate(imdb.classes):
        if cls == '__background__':
            continue        
        # print('Writing detection results file for class {}'.format(cls))
        for im_ind, index in enumerate(imdb.image_index):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            # the VOCdevkit expects 1-based indices
            for k in xrange(dets.shape[0]):
                if output_fn:
                    of.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f} {:s}\n'.
                      format(index, dets[k, -1],
                            dets[k, 0] + 1, dets[k, 1] + 1,
                            dets[k, 2] + 1, dets[k, 3] + 1,
                            num_recognitions[cls_ind][im_ind][k]))
                else:
                    print('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f} {:s}'.
                          format(index, dets[k, -1],
                                dets[k, 0] + 1, dets[k, 1] + 1,
                                dets[k, 2] + 1, dets[k, 3] + 1,
                                num_recognitions[cls_ind][im_ind][k]))
    if output_fn:
        of.close()

