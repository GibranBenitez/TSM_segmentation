#!/bin/bash 	#dl04
sup=""
host=$(hostname -s)

python main.py ipn RGB \
   --arch resnet50 --num_segments 8 \
   --gd 20 --lr 0.01 --lr_steps 40 80 --epochs 100 \
   --batch-size 32 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=2 \
   --shift --shift_div=8 --shift_place=blockres --npb --hostname $host$sup