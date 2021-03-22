#!/bin/bash

DIR=../checkpoints
FILE=scg_1e-4_b32h16e7_hicodet_e2e.pt
ID=1yOebBfiNIR20EyAZGap88ecUgMR8J0X_

if [ ! -d $DIR ]; then
   mkdir $DIR
fi 

if [ -f $DIR/$FILE ]; then
  echo "$FILE already exists."
  exit 0
fi

echo "Connecting..."

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&id=$ID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')&id=$ID" -O $FILE && rm -rf /tmp/cookies.txt

mv $FILE $DIR/

echo "Done."
