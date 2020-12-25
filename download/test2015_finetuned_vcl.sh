#!/bin/bash

DIR=../hicodet/detections
FILE=test2015_finetuned_vcl.tar.gz
ID=1eMj8DON8NkutD6kWT_U-xiP4B5jsKXF6

if [ -f $FILE ]; then
  echo "$FILE already exists."
  exit 0
fi

echo "Connecting..."

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&id=$ID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')&id=$ID" -O $FILE && rm -rf /tmp/cookies.txt

echo "Extracting..."

tar zxf $FILE

echo "Relocating and cleaning up..."

rm $FILE
mv test2015_finetuned_vcl $DIR/

echo "Done."
