for video in `find Sessions -name "*.avi"`
do
  target=$(basename $(dirname $video))
  keyframe-extractor -v $video -r frames -d $target -config ./config.txt
done