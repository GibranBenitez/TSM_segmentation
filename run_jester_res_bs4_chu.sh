#!/bin/bash 	#dl04
python main.py jester RGB \
   --arch resnet50 --num_segments 8 \
   --gd 20 --lr 0.01 --lr_steps 20 40 --epochs 50 \
   --batch-size 4 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
   --shift --shift_div=8 --shift_place=blockres --npb