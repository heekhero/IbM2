import os
import shutil

for root, dirs, files in os.walk('../checkpoint'):
    for dir in dirs:
        if ('pre_norm_True_opt_adam_search_lr_1.0_bs_512_fc_normlinear_tao_10.0' in dir):
            shutil.rmtree(os.path.join(root, dir))
