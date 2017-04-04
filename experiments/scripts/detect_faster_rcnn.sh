#!/bin/bash
#output/faster_rcnn_end2end/bib_500X500Gray_training/vgg16_faster_rcnn_iter_70000.caffemodel \
GPU_ID=$1
IMAGES=$2
FASTER_RCNN_NET=$3
FASTER_RCNN_NET_CKPT=$4
NUMBER_NET_CKPT=$5
TRAIN_IMDB="bib_500X500_training"
ITERS=70000
ANCHORS="[8,16,32]"
EXTRA_ARGS=${array[@]:6:$len}
CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/detect_net.py \
    --model ${FASTER_RCNN_NET_CKPT} ${NUMBER_NET_CKPT} \
    --source_files ${IMAGES} \
    --cfg experiments/cfgs/${FASTER_RCNN_NET}.yml \
    --faster_rcnn_net ${FASTER_RCNN_NET} \
    --set ANCHOR_SCALES ${ANCHORS} ${EXTRA_ARGS}
  