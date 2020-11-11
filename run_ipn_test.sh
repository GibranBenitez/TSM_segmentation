#!/bin/bash 	#dl04

python test_models_own.py \
   --dataset ipn --test_segments "8" \
   --weights ./checkpoint/TSM_ipn14_RGB-seg_resnet50_shift8_blockres_avg_segment8_e50_lr1e-03_gp30/ \
   --batch_size 32 -j 16 --dense_sample
