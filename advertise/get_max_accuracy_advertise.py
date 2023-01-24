import os
from collections import defaultdict
import sys
import init_path

from config import PATH

def is_float(x):
    try:
        float(x)
        return True
    except:
        return False

for root, dirs, files in os.walk(os.path.join(PATH, 'checkpoint')):
    accs_naive = {}
    accs_bsearch = {}
    for file in files:
        if ('naive_bsearch' == root.split('/')[-1]) and ('channel_wise_with_train_aug_search_lr_0.1_bs_256_fc_normlinear_tao_10.0_cri_smooth_rd_0.95_M_200_Pre_100_C_100_F_100' in root.split('/')) and ('max_results' not in file):
            with open(os.path.join(root, file), 'r') as f:
                lines = f.readlines()
                args_dict = {}
                args_pair = lines[0].replace('\'', '').replace('{', '').replace('}', '').split(',')

                for pair in args_pair:
                    k, v = pair.split(':')
                    k = k.strip()
                    v = v.strip()
                    args_dict[k] = v
                file_path = os.path.abspath(os.path.join(root, file))


                try:
                    assert (args_dict['log_file'].split('checkpoint')[-1] == file_path.split('checkpoint')[-1].replace('naive_bsearch/', '')) or (args_dict['log_file'].split('checkpoint')[-1] == file_path.split('checkpoint')[-1])
                except:
                    print(args_dict['log_file'])
                    print(file_path)
                    sys.exit(-1)


                line_list = lines[-1].strip().split(' ')
                try:
                    lr = line_list[8]
                    naive_acc_mean = line_list[16]
                    naive_acc_std = line_list[18]
                    bsearch_acc_mean = line_list[26]
                    bsearch_acc_std = line_list[28]

                    assert is_float(lr)
                    assert is_float(naive_acc_mean)
                    assert is_float(naive_acc_std)
                    assert is_float(bsearch_acc_mean)
                    assert is_float(bsearch_acc_std)

                    accs_naive[lr] = (naive_acc_mean, naive_acc_std)
                    accs_bsearch[lr] = (bsearch_acc_mean, bsearch_acc_std)

                except:
                    print(os.path.join(root, file) + '              might not finished...')
    if len(accs_naive) != 0:
        assert len(accs_naive) == len(accs_bsearch)
        naive_best_mean, naive_best_std = max(accs_naive.values(), key=lambda x : float(x[0]))
        bsearch_best_mean, bsearch_best_std = max(accs_bsearch.values(), key=lambda x: float(x[0]))

        with open(os.path.join(root, 'max_results_channel_wise_epsilon.txt'), 'w') as f:
            lrs = []
            lrs_key = []
            for lr in accs_naive:
                lrs_key.append(lr)
                lrs.append(float(lr))
            lrs = sorted(lrs)
            lrs_str = ','.join([str(lr) for lr in lrs])

            for lr in lrs_key:
                _acc_mean_naive, _acc_std_naive = accs_naive[lr]
                f.write('{:<10}     Lr : {}     Acc : {}+-{}\n'.format('Naive', lr, _acc_mean_naive, _acc_std_naive))
            f.write('\n')

            for lr in lrs_key:
                _acc_mean_bsearch, _acc_std_bsearch = accs_bsearch[lr]
                f.write('{:<10}     Lr : {}     Acc : {}+-{}\n'.format('BSearch', lr, _acc_mean_bsearch, _acc_std_bsearch))

            f.write('\n')
            f.write('***Final Results***   Among {}      Naive : {}+-{}      BSearch : {}+-{}\n'.format(lrs_str, naive_best_mean, naive_best_std, bsearch_best_mean, bsearch_best_std))
    
    