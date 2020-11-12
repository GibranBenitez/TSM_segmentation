#!/bin/bash 	#dl04

python test_models_own.py \
   --dataset ipn --test_segments "8" \
   --weights ./checkpoint/TSM_ipn14_RGB_resnet50_shift8_blockres_avg_segment8_e40_lr5e-04_tuned_gp30/ \
   --batch_size 32 -j 16 --dense_sample

#!/bin/bash 	#dl04

python test_models_own.py \
   --dataset ipn --test_segments "8" \
   --weights ./checkpoint/TSM_ipn14_RGB-seg_resnet50_shift8_blockres_avg_segment8_e40_lr5e-04_tuned_gp23/ \
   --batch_size 32 -j 16 --dense_sample

#!/bin/bash 	#dl04

python test_models_own.py \
   --dataset ipn --test_segments "8" \
   --weights ./checkpoint/TSM_ipn14_RGB-flo_resnet50_shift8_blockres_avg_segment8_e40_lr5e-04_tuned_gp30/ \
   --batch_size 32 -j 16 --dense_sample

#!/bin/bash 	#dl04

python test_models_own.py \
   --dataset ipn --test_segments "8" \
   --weights ./checkpoint/TSM_ipn14_RGB_resnet50_shift8_blockres_avg_segment8_e50_lr1e-03_gp30/ \
   --batch_size 32 -j 16 --dense_sample

#!/bin/bash 	#dl04

python test_models_own.py \
   --dataset ipn --test_segments "8" \
   --weights ./checkpoint/TSM_ipn14_RGB-flo_resnet50_shift8_blockres_avg_segment8_e50_lr1e-03_gp30/ \
   --batch_size 32 -j 16 --dense_sample

# #!/bin/bash 	#dl04

# python test_models_own.py \
#    --dataset ipn --test_segments "8" \
#    --weights ./checkpoint/TSM_ipn14_RGB-seg_resnet50_shift8_blockres_avg_segment8_e50_lr1e-03_gp30/ \
#    --batch_size 32 -j 16 --dense_sample

#!/bin/bash 	#dl04

python test_models_own.py \
   --dataset ipn --test_segments "8" \
   --weights ./checkpoint/TSM_ipn14_RGB_resnet50_avg_segment8_e50_lr1e-03_gp23/ \
   --batch_size 32 -j 16 --dense_sample

#!/bin/bash 	#dl04

python test_models_own.py \
   --dataset ipn --test_segments "8" \
   --weights ./checkpoint/TSM_ipn14_RGB-flo_resnet50_avg_segment8_e50_lr1e-03_gp23/ \
   --batch_size 32 -j 16 --dense_sample

#!/bin/bash 	#dl04

python test_models_own.py \
   --dataset ipn --test_segments "8" \
   --weights ./checkpoint/TSM_ipn14_RGB-seg_resnet50_avg_segment8_e50_lr1e-03_gp23/ \
   --batch_size 32 -j 16 --dense_sample

#!/bin/bash 	#dl04

python test_models_own.py \
   --dataset ipn --test_segments "16" \
   --weights ./checkpoint/TSM_ipn14_RGB-seg_resnet50_shift8_blockres_avg_segment16_e60_lr1e-03_gp30/ \
   --batch_size 16 -j 16 --dense_sample


#!/bin/bash 	#dl04

python test_models_own.py \
   --dataset ipn --test_segments "16" \
   --weights ./checkpoint/TSM_ipn14_RGB_resnet50_shift8_blockres_avg_segment16_e60_lr1e-03_gp30/ \
   --batch_size 16 -j 16 --dense_sample

#!/bin/bash 	#dl04

python test_models_own.py \
   --dataset ipn --test_segments "32" \
   --weights ./checkpoint/TSM_ipn14_RGB-seg_resnet50_shift8_blockres_avg_segment32_e60_lr1e-04_gp34/ \
   --batch_size 8 -j 8 --dense_sample


#!/bin/bash 	#dl04

python test_models_own.py \
   --dataset ipn --test_segments "32" \
   --weights ./checkpoint/TSM_ipn14_RGB_resnet50_shift8_blockres_avg_segment32_e60_lr1e-04_gp34/ \
   --batch_size 8 -j 8 --dense_sample

#!/bin/bash 	#dl04

python test_models_own.py \
   --dataset ipn --test_segments "24" \
   --weights ./checkpoint/TSM_ipn14_RGB-seg_resnet50_shift8_blockres_avg_segment24_e60_lr1e-03_gp30/ \
   --batch_size 8 -j 8 --dense_sample


#!/bin/bash 	#dl04

python test_models_own.py \
   --dataset ipn --test_segments "24" \
   --weights ./checkpoint/TSM_ipn14_RGB_resnet50_shift8_blockres_avg_segment24_e60_lr1e-03_gp23/ \
   --batch_size 8 -j 8 --dense_sample