#!/bin/bash

outfolder='../fits/'
encoder_name='starnet-m2'
# encoder_name='tmp'

python train_sleep.py \
  --outfolder $outfolder \
  --outfilename $encoder_name
