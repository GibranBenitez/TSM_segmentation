#!/bin/bash     #dl04
sup=""
host=$(hostname -s)
st=2
en=22
id=""
for (( i=$st; i<=$en; i++ ))
do
    id=$id" "`expr $i \* 1`
done  

echo "Training BioVid TSM from "$st" to "$en" val_IDs, in "$host$sup" server"
for val_id_ in $id
do
	echo " "
	echo " "
    echo "Runing with val ID "$val_id_" ..."
    
    python main.py biovid RGB \
       --arch resnet50 --num_segments 16 --bio_validation $val_id_\
       --lr 0.0001 --lr_steps 20 40 --epochs 50 --ipn_no_class 2\
       --batch-size 16 -j 8 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
       --shift --shift_div=8 --shift_place=blockres --hostname $host$sup

done