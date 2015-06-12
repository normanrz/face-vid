#!/usr/bin/env bash

export CAFFE_ROOT="$HOME/caffe-bartzi"

WEIGHTS=$CAFFE_ROOT/models/bvlc_alexnet/bvlc_alexnet.caffemodel

$CAFFE_ROOT/build/tools/caffe train \
    -solver /home/mpss2015/face-vid/face-vid-nets/flow/solver.prototxt 2>&1 | tee mmi-oao-flow.tlog | less
    # -gpu 0
#    -weights $WEIGHTS 2>&1 | tee mmi-oao-flow.tlog | less