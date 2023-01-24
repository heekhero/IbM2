import os
import subprocess
import numpy as np
import time
import sys
import argparse
import random

gpu_capacity = 10480

def get_available_gpu(logic_ids = [0,1,2,3,4,5,6,7], physical_ids = [0,1,2,3,4,5,6,7]):
    os.system('nvidia-smi -i {} -q -d Memory |grep -A4 GPU|grep Free >tmp_pascal1_etf'.format(','.join([str(_id) for _id in logic_ids])))
    memory_gpu = np.array([int(x.split()[2]) for x in open('tmp_pascal1_etf', 'r').readlines()])
    # free_gpu = np.where(np.array(memory_gpu) > gpu_capacity)[0]
    sorted_indices = np.argsort(-memory_gpu)
    free_gpu = [physical_ids[logic_ids[gid]] for gid in sorted_indices if memory_gpu[gid] > gpu_capacity]
    return free_gpu


if __name__ == '__main__':

    my_env = os.environ.copy()

    child_list = []
    parallel_size = 8

    logical_ids = [5, 6]
    physical_ids = [0,1,2,3,4,5,6,7]

    shots = ['4']
    archs = ['deit_large_p7']
    ts = ['0', '1']

    for arch in archs:
        for shot in shots:
            for _t in ts:
                free_gpus = get_available_gpu(logical_ids, physical_ids)

                while (len(free_gpus) == 0) or (len(child_list) >= parallel_size):
                    # print('wait')
                    time.sleep(60)
                    free_gpus = get_available_gpu(logical_ids, physical_ids)

                    for cd in child_list:
                        if cd.poll() is not None:
                            child_list.remove(cd)

                launch_id = str(free_gpus[0])
                print('launch in {}'.format(launch_id))
                my_env['CUDA_VISIBLE_DEVICES'] = launch_id
                with open('out/extract_features_fsl_distributed/{}_{}shot_t_{}.out'.format(arch, shot, _t), 'w') as f:
                    base_list = ['python', 'advertise/extract_features_aug100times_fsl_distributed.py', '--shot', shot, '--arch', arch, '--eps', '25', '--batch_size', '32']

                    child = subprocess.Popen(
                        base_list,
                        env=my_env,
                        stdout=f,
                        stderr=f,
                        preexec_fn=os.setsid
                    )
                time.sleep(60)

