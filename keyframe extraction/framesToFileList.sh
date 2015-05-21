#!/bin/bash

Output="filelist.txt"
if [ -f $Output ]; then
  rm $Output
fi

for frameFile in `find . -name "frame*.png"`
do
  fileOnly=$(basename $frameFile)
  fileWithoutEnding=${fileOnly%.png}
  labels=${fileWithoutEnding#frame-}
  chosenLabel=$(echo $labels | cut -d_ -f1)

  fullPath=$(readlink -f $frameFile)
  echo $fullPath $chosenLabel >> $Output
done
~       
