#!/usr/bin/env bash

export CAFFE_ROOT="$HOME/caffe-tmbo"

$CAFFE_ROOT/build/tools/caffe train \
    -solver /home/mpss2015/face-vid/face-vid-nets/flow/solver.prototxt 2>&1 | tee logs/mmi-oao-flow.tlog | less
