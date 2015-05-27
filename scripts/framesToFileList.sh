find frames -type f -iname 'frame-bgr*' | sed -r 's/^[^_]*_([0-9]+)[\._].*$/\0 \1/g' >> files.txt

