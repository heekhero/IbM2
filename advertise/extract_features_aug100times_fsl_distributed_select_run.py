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
    parser.add_argument('--shot', default=1, type=int)
    parser.add_argument('--run', default=0, type=int)

    parser.add_argument('--eps', default=20, type=int)
    parser.add_argument('--arch', default='deit_large_p7', choices=['deit_large_p7', 'deit_base_p4'])
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--pretrain_method', default='MSN', choices=['MSN'])
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--checkpoint', default='checkpoint')

    args = parser.parse_args()

    args.exp_dir = os.path.join(args.checkpoint, args.pretrain_method, args.dataset, args.arch, 'features')

    if args.pretrain_method == 'MSN':
        if args.arch == 'deit_large_p7':
            args.load_path = 'checkpoint/MSN/vitl7_200ep.pth.tar'
        elif args.arch == 'deit_base_p4':
            args.load_path = 'checkpoint/MSN/vitb4_300ep.pth.tar'
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    print(args)

    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    mset = FewShotMetaSet(setname=args.dataset, shot=args.shot)

    if args.pretrain_method == 'MSN':
        if args.arch == 'deit_large_p7':
            model = deit_large_p7()
        elif args.arch == 'deit_base_p4':
            model = deit_base_p4()
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    print(model)

    load_dict = torch.load(args.load_path, map_location=torch.device('cpu'))
    load_dict = {k.replace('module.', ''): v for k, v in load_dict['target_encoder'].items() if k.replace('module.', '') in model.state_dict()}

    print('len of load dict {}'.format(len(load_dict)))

    msg = str(model.load_state_dict(load_dict, strict=False))
    print(msg)
    assert '<All keys matched successfully>' in msg

    model.cuda()
    model.eval()

    trains_set = mset.trains
    rid = args.run
    train_set = trains_set[rid]
    dloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers)
    save_dir = os.path.join(args.exp_dir, 'aug_100times', '{}shot'.format(args.shot), 'run_{}'.format(rid))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for ep in range(args.eps):
            feature_list = []
            target_list = []
            for images, targets in tqdm(dloader):

                images = images.float().cuda(non_blocking=True)

                features = model(images).cpu()

                feature_list.append(features.detach().cpu())
                target_list.append(targets.detach().cpu())


            features_gather = torch.cat(feature_list, dim=0)
            cyan.preprocess(features_gather, normalize=True, columns=False, centering=True)
            targets_gather = torch.cat(target_list, dim=0)


            feature_dict = {}
            feature_dict['features'] = features_gather
            feature_dict['targets'] = targets_gather
            print('len of features', len(features_gather))

            rand_d = np.random.randint(0, 65535)
            save_path = os.path.join(save_dir, 'train_ep_{}.pth'.format(rand_d))
            with open(save_path, 'wb') as f:
                pickle.dump(feature_dict, f)




if __name__ == '__main__':
    main()