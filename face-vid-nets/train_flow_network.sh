#!/usr/bin/env bash

export CAFFE_ROOT="$HOME/caffe-tmbo"

WEIGHTS=$CAFFE_ROOT/models/bvlc_alexnet/bvlc_alexnet.caffemodel

$CAFFE_ROOT/build/tools/caffe train \
    -solver $CAFFE_ROOT/examples/face-vid-nets/flow/solver.prototxt \
    -weights $WEIGHTS 2>&1 | tee mmi-oao-flow.tlog | less
#    -gpu 0 2>&1 | less