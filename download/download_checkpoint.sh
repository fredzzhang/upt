#!/bin/bash

DIR=../checkpoints
FILE=spatio-attentive-graphs-hicodet-e11.pt
ID=1tAkqKx_IIcV_IOxKKE8QYTBZLvEqsZYX

if [ ! -d $DIR ]; then
   mkdir $DIR
fi 

if [ -f $DIR/$FILE ]; then
  echo "$FILE already exists."
  exit 0
fi

echo "Connecting..."

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&id=$ID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')&id=$ID" -O $FILE && rm -rf /tmp/cookies.txt

echo "Done."
