#!/bin/bash 	#dl04
export CUDA_VISIBLE_DEVICES=0,1,2
sup=""
host=$(hostname -s)

python main.py ipn RGB \
   --arch resnet50 --num_segments 8 \
   --gd 20 --lr 0.01 --lr_steps 40 80 --epochs 100 \
   --batch-size 32 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=2 \
   --shift --shift_div=8 --shift_place=blockres --npb --hostname $host$sup

#!/bin/bash     #dl04
sup=""
host=$(hostname -s)

python main.py ipn RGB \
   --arch resnet50 --num_segments 8 \
   --lr 0.001 --lr_steps 20 40 --epochs 50 --ipn_no_class 1\
   --batch-size 32 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=2 \
   --shift --shift_div=8 --shift_place=blockres --hostname $host$sup
############ IPN tunned
#!/bin/bash     #dl04
sup=""
host=$(hostname -s)

python main.py ipn RGB-flo \
   --arch resnet50 --num_segments 8 \
   --lr 0.0005 --lr_steps 15 30 --epochs 40 --ipn_no_class 1 \
   --batch-size 32 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=2 \
   --shift --shift_div=8 --shift_place=blockres --hostname $host$sup --suffix tuned \
   --tune_from=checkpoint/TSM_jester_RGB_resnet50_shift8_blockres_avg_segment8_e50/ckpt.best.pth.tar

########## Last IPN models
#!/bin/bash     #dl04 TSN NO TSM
sup=""
host=$(hostname -s)

python main.py ipn RGB-seg \
   --arch resnet50 --num_segments 8 \
   --lr 0.001 --lr_steps 20 40 --epochs 50 --ipn_no_class 4\
   --batch-size 32 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=2 \
   --hostname $host$sup

#!/bin/bash     #dl04
sup=""
host=$(hostname -s)

python main.py ipn RGB-seg \
   --arch resnet50 --num_segments 8 \
   --lr 0.001 --lr_steps 20 40 --epochs 50 --ipn_no_class 4\
   --batch-size 32 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=2 \
   --shift --shift_div=8 --shift_place=blockres --hostname $host$sup

#!/bin/bash     #dl04
sup=""
host=$(hostname -s)

python main.py ipn RGB-flo \
   --arch resnet50 --num_segments 8 \
   --lr 0.001 --lr_steps 20 40 --epochs 50 --ipn_no_class 2\
   --batch-size 32 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=2 \
   --shift --shift_div=8 --shift_place=blockres --hostname $host$sup

#!/bin/bash     #dl04
sup=""
host=$(hostname -s)

python main.py ipn RGB-seg \
   --arch resnet50 --num_segments 16 \
   --lr 0.001 --lr_steps 20 40 --epochs 60 --ipn_no_class 4\
   --batch-size 16 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=2 \
   --shift --shift_div=8 --shift_place=blockres --hostname $host$sup

#!/bin/bash     #dl04
sup=""
host=$(hostname -s)

python main.py ipn RGB \
   --arch resnet50 --num_segments 24 \
   --lr 0.001 --lr_steps 20 40 --epochs 60 --ipn_no_class 4\
   --batch-size 12 -j 12 --dropout 0.5 --consensus_type=avg --eval-freq=2 \
   --shift --shift_div=8 --shift_place=blockres --hostname $host$sup

#!/bin/bash     #dl04
sup=""
host=$(hostname -s)

python main.py ipn RGB-seg \
   --arch resnet50 --num_segments 32 \
   --lr 0.0001 --lr_steps 20 40 --epochs 60 --ipn_no_class 4\
   --batch-size 24 -j 8 --dropout 0.5 --consensus_type=avg --eval-freq=2 \
   --shift --shift_div=8 --shift_place=blockres --hostname $host$sup
