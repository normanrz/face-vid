#!/bin/bash

if [ $# -lt 2 ]; then
  echo "Usage: $0 <fullpath to hdf5files>"
  exit 1
fi

declare -a streams=(flows framesBGR)
declare -a modes=(train test)

for stream in ${streams[@]}; do
  for mode in ${modes[@]}; do
    ls -1 $1/$stream_$mode* > $stream_$mode_source.txt 
  done
done
