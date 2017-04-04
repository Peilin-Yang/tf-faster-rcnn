#!/bin/bash
#output/faster_rcnn_end2end/bib_500X500Gray_training/vgg16_faster_rcnn_iter_70000.caffemodel \
GPU_ID=$1
NET=$2
TRAIN_IMDB="bib_500X500_training"
ITERS=70000
NET_FINAL=output/${NET}/${TRAIN_IMDB}/default/${NET}_faster_rcnn_iter_${ITERS}.ckpt
ANCHORS="[8,16,32]"
EXTRA_ARGS=${array[@]:3:$len}
CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/detect_net.py \
    --model ${NET_FINAL} \
    --cfg experiments/cfgs/${NET}.yml \
    --faster_rcnn_net ${NET} \
    --set ANCHOR_SCALES ${ANCHORS} ${EXTRA_ARGS}
  