#!/usr/bin/env bash

export CAFFE_ROOT="$HOME/caffe-tmbo"
export LD_LIBRARY_PATH=$CAFFE_ROOT/build/lib:$LD_LIBRARY_PATH

WEIGHTS=$CAFFE_ROOT/models/bvlc_alexnet/bvlc_alexnet.caffemodel

$CAFFE_ROOT/build/tools/caffe train \
    -solver $CAFFE_ROOT/examples/face-vid-nets/an-finetune/solver.prototxt \
    -weights $WEIGHTS > logs/mmi-oao-an.tlog 2>&1
#    -gpu 0 2>&1 | less
