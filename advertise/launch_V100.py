import os
import subprocess
import numpy as np
import time
import sys
import argparse
import random

gpu_capacity = 30000

def get_available_gpu(logic_ids = [0,1,2,3,4,5,6,7], physical_ids = [0,1,2,3,4,5,6,7]):
    os.system('nvidia-smi -i {} -q -d Memory |grep -A4 GPU|grep Free >tmp_V100_etf_aug'.format(','.join([str(_id) for _id in logic_ids])))
    memory_gpu = np.array([int(x.split()[2]) for x in open('tmp_V100_etf_aug', 'r').readlines()])
    # free_gpu = np.where(np.array(memory_gpu) > gpu_capacity)[0]
    sorted_indices = np.argsort(-memory_gpu)
    free_gpu = [physical_ids[logic_ids[gid]] for gid in sorted_indices if memory_gpu[gid] > gpu_capacity]
    return free_gpu


if __name__ == '__main__':

    my_env = os.environ.copy()

    parallel_size = 8

    logical_ids = [0,1,2,3,4,5,6,7]
    physical_ids = [0,1,2,3,4,5,6,7]

    shots = ['5']

    for shot in shots:
        free_gpus = get_available_gpu(logical_ids, physical_ids)

        while (len(free_gpus) < parallel_size):
            # print('wait')
            time.sleep(300)
            free_gpus = get_available_gpu(logical_ids, physical_ids)

        launch_id = ','.join([str(i) for i in free_gpus[:parallel_size]])
        print('launch in {}'.format(launch_id))
        with open('out/bsearch_finetune_train_aug/{}shot_.out'.format(shot), 'w') as f:
            base_list = ['bash', 'advertise/bsearch_finetune_channel_wise_train_aug.sh', launch_id, shot, 'Imagenet', 'deit_large_p7', 'MSN', '0.95', 'normlinear', '10.0', '0.1', '200']

            child = subprocess.Popen(
                base_list,
                env=my_env,
                stdout=f,
                stderr=f,
                preexec_fn=os.setsid
            )
        time.sleep(180)

