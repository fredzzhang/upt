#!/bin/bash

DATADIR=../hicodet/detections

bash test2015_finetuned_drg.sh
bash test2015_finetuned_vcl.sh
bash test2015_finetuned_ours.sh

mv test2015_finetuned_drg $DATADIR/
mv test2015_finetuned_vcl $DATADIR/
mv test2015_finetuned_ours $DATADIR/
