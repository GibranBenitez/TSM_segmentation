#!/bin/bash     #dl04
sup=""
host=$(hostname -s)

python main.py biovid RGB \
   --arch resnet50 --num_segments 16 --bio_validation 1\
   --lr 0.0001 --lr_steps 20 40 --epochs 50 --ipn_no_class 2\
   --batch-size 16 -j 8 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
   --shift --shift_div=8 --shift_place=blockres --hostname $host$sup
