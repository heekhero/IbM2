#/bin/bash
cuda_id=$1
shot=$2
dtset=$3
arch=$4
pretrain_method=$5
rd=$6
fc=$7
tao=$8
search_lr=$9
M=${10}


function Listening {
  TCPListening=`netstat -an | grep ":$1" | awk '/^tcp.*/ && $NF == "LISTEN" {print $0}'| wc -l`
  UDPListening=`netstat -an | grep ":$1" | awk '/^udp.*/ && $NF == "0.0.0.0:*" {print $0}'| wc -l`
  ((Listening=TCPListening+UDPListening))
  if [ $Listening == 0 ]; then
    echo 0
  else
    echo 1
  fi
}

function get_random_port {
  PORT=0
  while [ $PORT == 0 ]; do
    temp1=`shuf -i 12000-13000 -n1`
    if [ `Listening $temp1` == 0 ]; then
      PORT=$temp1
    fi
  done
  echo $PORT
}

CUDA_VISIBLE_DEVICES=$cuda_id python -m torch.distributed.launch --nproc_per_node 8 --master_port `get_random_port` advertise/bsearch_decouple_channel_wise_train_aug.py --shot $shot --dataset $dtset --arch $arch --pretrain_method $pretrain_method  --round $rd --fc $fc --scale_factor $tao --search_lr $search_lr --M $M --batch_size_per_gpu 32 --batch_size_test_per_gpu 32
for lr in 0.005 0.01 0.05 0.1 0.5 1.0 5.0; do
  CUDA_VISIBLE_DEVICES=$cuda_id python -m torch.distributed.launch --nproc_per_node 8 --master_port `get_random_port` advertise/finetune_decouple_channel_wise_train_aug.py --shot $shot --dataset $dtset --arch $arch --pretrain_method $pretrain_method --round $rd  --fc $fc --scale_factor $tao --search_lr $search_lr --M $M --batch_size_per_gpu 32 --batch_size_test_per_gpu 32   --train_lr $lr
done


