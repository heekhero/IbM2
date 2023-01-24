#/bin/bash
shot=$1
dtset=$2
arch=$3
pretrain_method=$4
rd=$5
fc=$6
tao=$7
search_lr=$8
M=${9}

for lr in 0.005 0.01 0.05 0.1 0.5 1.0 5.0; do
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12661 finetune_decouple_channel_wise_train_aug.py --shot $shot --dataset $dtset --arch $arch --pretrain_method $pretrain_method --round $rd  --fc $fc --scale_factor $tao --search_lr $search_lr --M $M --batch_size_per_gpu 32 --batch_size_test_per_gpu 32   --train_lr $lr
done


