if [ ! -d frames ]; then
  mkdir frames
fi

for video in `find Sessions -name "*.avi"`
do
  target='frames/'$(basename $(dirname $video))
  if [ -d $target ]; then
    rm -rf $target
  fi
  mkdir $target
  echo $target
  python parse_mmi_xml.py $video $target
done
