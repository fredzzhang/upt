#!/bin/bash

FILE=test2015_finetuned_drg.tar.gz
ID=1UCrgUgF1MSDjAcunqHpszOUDYCgfdOHw

if [ -f $FILE ]; then
  echo "$FILE already exists."
  exit 0
fi

echo "Connecting..."

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&id=$ID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')&id=$ID" -O $FILE && rm -rf /tmp/cookies.txt

echo "Extracting..."

tar zxf $FILE

echo "Done."
