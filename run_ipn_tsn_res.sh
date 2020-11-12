#!/bin/bash     #dl04
sup=""
host=$(hostname -s)

python main.py ipn RGB-flo \
   --arch resnet50 --num_segments 32 \
   --lr 0.0001 --lr_steps 20 40 --epochs 60 --ipn_no_class 1\
   --batch-size 24 -j 8 --dropout 0.5 --consensus_type=avg --eval-freq=2 \
   --hostname $host$sup

#!/bin/bash     #dl04
sup=""
host=$(hostname -s)

python main.py ipn RGB-flo \
   --arch resnet50 --num_segments 32 \
   --lr 0.0001 --lr_steps 20 40 --epochs 60 --ipn_no_class 4\
   --batch-size 24 -j 8 --dropout 0.5 --consensus_type=avg --eval-freq=2 \
   --hostname $host$sup