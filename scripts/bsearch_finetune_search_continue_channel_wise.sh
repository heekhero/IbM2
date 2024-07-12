#/bin/bash
cuda_id=$1
shot=$2
dtset=$3
arch=$4
pretrain_method=$5

if [ "$arch" = "resnet50" ]; then
  epochs=60
  batch_size=512
else
  epochs=100
  batch_size=256
fi

CUDA_VISIBLE_DEVICES=$cuda_id python bsearch_decouple_search_continue_channel_wise.py --shot $shot --dataset $dtset --arch $arch --pretrain_method $pretrain_method --pre_epochs $epochs --final_epochs $epochs --batch_size_per_gpu $batch_size
CUDA_VISIBLE_DEVICES=$cuda_id python finetune_decouple_search_continue_channel_wise.py --shot $shot --dataset $dtset --arch $arch --pretrain_method $pretrain_method  --pre_epochs $epochs --final_epochs $epochs --batch_size_per_gpu $batch_size


