#!/bin/bash

DB_NAME="MMI_OAO"
FILELIST="frames/filelist.txt"

# Collect all Frames as Filelist
if [ -f $FILELIST ]; then
  rm $FILELIST
fi

#find frames -type f -iname 'frame-bgr*' | sed -r 's/^[^_]*_([0-9]+[A-Za-z]*)[\._].*$/\0 \1/g' >> $FILELIST
python framesToFile_OAO.py frames

# Write LMDB
if [ -d $DB_NAME ]; then
  rm -rf $DB_NAME
fi

$CAFFE_ROOT/build/tools/convert_imageset /opt/data_sets/mmi_oao_facs/ $FILELIST $DB_NAME --logtostderr=1

