import os
import subprocess
import numpy as np
import time
import sys
import argparse
import random
import socket

HOST_NAME = socket.getfqdn(socket.gethostname())

if 'ubuntu' in HOST_NAME:
    prefix = 'root'
else:
    prefix = 'opt'

gpu_capacity = 5600

def get_available_gpu(logic_ids = [0,1,2,3,4,5,6,7], physical_ids = [0,1,2,3,4,5,6,7]):
    os.system('nvidia-smi -i {} -q -d Memory |grep -A4 GPU|grep Free >tmp_{}'.format(','.join([str(_id) for _id in logic_ids]), HOST_NAME))
    memory_gpu = np.array([int(x.split()[2]) for x in open('tmp_{}'.format(HOST_NAME), 'r').readlines()])
    sorted_indices = np.argsort(-memory_gpu)
    free_gpu = [physical_ids[logic_ids[gid]] for gid in sorted_indices if memory_gpu[gid] > gpu_capacity]
    return free_gpu


if __name__ == '__main__':

    my_env = os.environ.copy()

    child_list = []
    parallel_size = 8

    logical_ids = [0,1,2,3,4,5,6,7]
    physical_ids = [0,1,2,3,4,5,6,7]

    datasets = ['Imagenet', 'CUB']
    pretrain_methods = ['SimCLR', 'BYOL', 'SwAV', 'DINO', 'MSN', 'MoCov3', 'iBOT']
    shots = ['1', '2', '3']
    rights = ['20.0']
    batch_sizes_search = ['256']
    pre_epochs = ['100']
    rds = ['0.95']

    for shot in shots:
        for dataset in datasets:
            for pretrain_method in pretrain_methods:
                if pretrain_method == 'DINO':
                    archs = ['deit_small_p8', 'deit_small_p16']
                elif pretrain_method == 'MSN':
                    archs = ['deit_base_p4', 'deit_large_p7', 'deit_small_p16']
                elif pretrain_method in ['SimCLR', 'BYOL', 'SwAV']:
                    archs = ['resnet50']
                else:
                    archs = ['deit_small_p16']


                if pretrain_method in ['SimCLR', 'BYOL', 'SwAV']:
                    final_epoch = '60'
                    batch_size_train = '512'
                else:
                    final_epoch = '100'
                    batch_size_train = '256'

                for arch in archs:
                    for batch_size_search in batch_sizes_search:
                        for pre_epoch in pre_epochs:
                            for rd in rds:
                                for right in rights:
                                    free_gpus = get_available_gpu(logical_ids, physical_ids)

                                    while (len(free_gpus) == 0) or (len(child_list) >= parallel_size):
                                        # print('wait')
                                        time.sleep(300)
                                        free_gpus = get_available_gpu(logical_ids, physical_ids)

                                        for cd in child_list:
                                            if cd.poll() is not None:
                                                child_list.remove(cd)

                                    launch_id = str(free_gpus[0])
                                    print('launch in {}'.format(launch_id))
                                    with open('out/{}_{}_{}_pre_epochs_{}_final_epochs_{}_bs_search_{}_bs_train_{}_rd_{}_{}shot_right_{}.out'.format(dataset, pretrain_method, arch, pre_epoch, final_epoch, batch_size_search, batch_size_train, rd, shot, right), 'w') as f:
                                        base_list = ['bash', 'scripts/bsearch_finetune_search_continue_channel_wise.sh', launch_id, shot, dataset, arch, pretrain_method, rd, pre_epoch, final_epoch, batch_size_search, batch_size_train, right]

                                        child = subprocess.Popen(
                                            base_list,
                                            env=my_env,
                                            stdout=f,
                                            stderr=f,
                                        )
                                        child_list.append(child)
                                        time.sleep(120)

