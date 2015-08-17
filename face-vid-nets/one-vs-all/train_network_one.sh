#!/usr/bin/env bash

export CAFFE_ROOT="$HOME/caffe-tmbo"

WEIGHTS=$CAFFE_ROOT/models/bvlc_alexnet/bvlc_alexnet.caffemodel

~/caffe-tmbo/build/tools/caffe train \
    -solver $1 \
    -weights $WEIGHTS > logs/mmi-oao-one-$2.tlog 2>&1
#    -gpu 0 2>&1 | less