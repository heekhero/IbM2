#/bin/bash
cuda_id=$1
shot=$2
dtset=$3
arch=$4
pretrain_method=$5
rd=$6
pre_epochs=$7
final_epochs=$8
batch_size_search=$9
batch_size_train=${10}
right=${11}

CUDA_VISIBLE_DEVICES=$cuda_id python bsearch_decouple_search_continue_channel_wise.py --shot $shot --dataset $dtset --arch $arch --pretrain_method $pretrain_method  --round $rd --pre_epochs $pre_epochs --final_epochs $final_epochs --batch_size_per_gpu_search $batch_size_search --batch_size_per_gpu_train $batch_size_train --right $right
for lr in 0.001 0.005 0.01 0.05 0.1 0.5 1.0 5.0; do
  CUDA_VISIBLE_DEVICES=$cuda_id python finetune_decouple_search_continue_channel_wise.py --shot $shot --dataset $dtset --arch $arch --pretrain_method $pretrain_method --round $rd  --pre_epochs $pre_epochs --final_epochs $final_epochs --batch_size_per_gpu_search $batch_size_search --batch_size_per_gpu_train $batch_size_train --right $right  --train_lr $lr
done


