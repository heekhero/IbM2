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

gpu_capacity = 10240
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
    physical_ids = [0,1,2,3,4,7,5,6]

    pts = ['1']
    pretrain_methods = ['DINO', 'MSN', 'iBOT', 'MoCov3']

    for pt in pts:
        for pretrain_method in pretrain_methods:
            if pretrain_method == 'DINO':
                archs = ['deit_small_p8', 'deit_small_p16']
            elif pretrain_method == 'MSN':
                archs = ['deit_base_p4', 'deit_large_p7', 'deit_small_p16']
            else:
                archs = ['deit_small_p16']
            for arch in archs:
                if 'small' in arch:
                    batch_size = '128'
                elif 'large' in arch:
                    batch_size = '32'
                else:
                    batch_size = '4'
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
                my_env['CUDA_VISIBLE_DEVICES'] = launch_id
                with open('out/extract_features_{}_{}_{}pt.out'.format(arch, pretrain_method, pt), 'w') as f:
                    base_list = ['python', '-u', 'extract_features_imagenet_subsets.py', '--arch', arch, '--pretrain_method', pretrain_method, '--pt', pt, '--batch_size', batch_size]

                    child = subprocess.Popen(
                        base_list,
                        env=my_env,
                        stdout=f,
                        stderr=f,
                    )
                    child_list.append(child)
                    time.sleep(60)

