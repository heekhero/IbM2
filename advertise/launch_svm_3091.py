import os
import subprocess
import numpy as np
import time
import sys
import argparse
import random

gpu_capacity = 2048

def get_available_gpu(logic_ids = [0,1,2,3,4,5,6,7], physical_ids = [0,1,2,3,4,5,6,7]):
    os.system('nvidia-smi -i {} -q -d Memory |grep -A4 GPU|grep Free >tmp_3091_svc_system'.format(','.join([str(_id) for _id in logic_ids])))
    memory_gpu = np.array([int(x.split()[2]) for x in open('tmp_3091_svc_system', 'r').readlines()])
    # free_gpu = np.where(np.array(memory_gpu) > gpu_capacity)[0]
    sorted_indices = np.argsort(-memory_gpu)
    free_gpu = [physical_ids[logic_ids[gid]] for gid in sorted_indices if memory_gpu[gid] > gpu_capacity]
    return free_gpu


if __name__ == '__main__':

    my_env = os.environ.copy()
    # my_env["PATH"] = "/opt/fumh/miniconda3/envs/maml/bin:" + my_env["PATH"]

    shots = ['1', '2', '3', '4', '5', '8', '16']
    archs = ['deit_large_p7']
    Cs = ['1.0']

    for arch in archs:
        for shot in shots:
            for aug100times in [True, False]:
                for C in Cs:

                    with open('out/svc_system_search/svc_{}_{}shot_aug_{}_C_{}.out'.format(arch, shot, aug100times, C), 'w') as f:
                        base_list = ['python', 'advertise/svm_svc.py', '--shot', shot, '--arch', arch, '--C', C]

                        if aug100times:
                            base_list += ['--aug_100times']

                        child = subprocess.Popen(
                            base_list,
                            env=my_env,
                            stdout=f,
                            stderr=f,
                            preexec_fn=os.setsid
                        )
