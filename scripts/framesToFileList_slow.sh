#!/bin/bash

Output="filelist.txt"
if [ -f $Output ]; then
  rm $Output
fi

for framefile in `find frames -name "frame*.png"`
do
  fileOnly=$(basename $framefile)
  fileWithoutEnding=${fileOnly%.png}
  labels=${fileWithoutEnding#frame-}
  chosenLabel=$(echo $labels | cut -d_ -f1)
  
  fullPath=$(readlink -f $framefile)
  echo $fullPath $chosenLabel >> $Output
done
