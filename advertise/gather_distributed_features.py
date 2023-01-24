import argparse
import copy
import os
import pickle
import random
import time

import numpy as np
import torch.nn as nn
import torch.backends.cudnn
import torch.optim
import torch.utils.data
import init_path
from tqdm import tqdm
import cyanure as cyan

from collections import defaultdict
from model.msn_deit import deit_small, deit_large_p7, deit_base_p4
from model.dino_vision_transformer import vit_small
from model.mocov3_vits import vit_small as mocov3_vit_small
from model.iBOT_vision_transformer import vit_small as iBOT_vit_small
from config import PATH
from datasets.few_shot_dataset import FewShotMetaSet
from utils import *

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='Imagenet', choices=['Imagenet', 'Imagenet_1pt', 'Imagenet_10pt'])
    parser.add_argument('--shot', default=8, type=int)

    parser.add_argument('--arch', default='deit_large_p7', choices=['deit_large_p7', 'deit_base_p4'])
    parser.add_argument('--pretrain_method', default='MSN', choices=['MSN'])

    parser.add_argument('--checkpoint', default='checkpoint')

    args = parser.parse_args()

    args.exp_dir = os.path.join(PATH, args.checkpoint, args.pretrain_method, args.dataset, args.arch, 'features')

    print(args)

    all_features = defaultdict(list)
    all_targets = defaultdict(list)
    for rid in range(3):
        save_dir = os.path.join(args.exp_dir, 'aug_100times', '{}shot'.format(args.shot), 'run_{}'.format(rid))
        for root, dirs, files in os.walk(save_dir):
            for file in files:
                load_path_now = os.path.join(save_dir, file)
                with open(load_path_now, 'rb') as f:
                    feature_dict = pickle.load(f)
                    all_features['run_{}'.format(rid)].append(feature_dict['features'])
                    all_targets['run_{}'.format(rid)].append(feature_dict['targets'])

    for rid in range(3):
        r_features = all_features['run_{}'.format(rid)]
        r_targets = all_targets['run_{}'.format(rid)]

        r_features = torch.cat(r_features, dim=0)
        r_targets = torch.cat(r_targets, dim=0)


        assert r_features.size(0) == r_targets.size(0)
        assert (r_targets.reshape(-1, 1000, args.shot)[0, :, 0].unsqueeze(0).unsqueeze(-1) == r_targets.reshape(-1, 1000, args.shot)).sum().item() == r_features.size(0)

        assert r_features.size(0) == args.shot * 1000 * 100

        save_path = os.path.join(args.exp_dir, 'train_{}shot_{}_aug100times.pth'.format(args.shot, rid))

        feature_dict = {}
        feature_dict['features'] = r_features
        feature_dict['targets'] = r_targets
        print('len of features', len(r_features))
        with open(save_path, 'wb') as f:
            pickle.dump(feature_dict, f)

if __name__ == '__main__':
    main()