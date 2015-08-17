#!/usr/bin/env bash

export CAFFE_ROOT="$HOME/caffe-tmbo"

WEIGHTS=$CAFFE_ROOT/models/bvlc_alexnet/bvlc_alexnet.caffemodel

~/caffe-tmbo/build/tools/caffe train \
    -solver $CAFFE_ROOT/examples/face-vid-nets/simple-net/solver.prototxt \
    -weights $WEIGHTS 2>&1 | tee mmi-oao-simple.tlog | less
#    -gpu 0 2>&1 | less