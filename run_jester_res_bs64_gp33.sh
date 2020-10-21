#!/bin/bash 	#dl04
export CUDA_VISIBLE_DEVICES=0,1,2

python main.py jester RGB \
   --arch resnet50 --num_segments 8 \
   --gd 20 --lr 0.01 --lr_steps 20 40 --epochs 50 \
   --batch-size 64 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
   --shift --shift_div=8 --shift_place=blockres --npb \
   --resume /host/space0/gibran/scripts/TSM/checkpoint/TSM_jester_RGB_resnet50_shift8_blockres_avg_segment8_e50/ckpt_20.pth.tar