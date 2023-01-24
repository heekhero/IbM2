import argparse
import os
import pickle
import random

import numpy as np
import torch.backends.cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import copy
import time
import torch.nn as nn
import math
import torch.distributed as dist
import init_path

from tqdm import tqdm
from collections import defaultdict, OrderedDict
from model.msn_deit import deit_large_p7, deit_base_p4
from config import PATH
from model.classifier import get_classifier
from datasets.few_shot_dataset import FewShotMetaSet
from utils import accuracy, AvgMetric, write, setup_for_distributed, LabelSmoothingCrossEntropy, concat_all_gather

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.benchmark = True

def train_loop(args, classifier, backbone, train_loader, lr, epsilon, M, epochs, test_loader=None, flag='training'):
    write('Starting {} from epsilon={:.4f} with M={}'.format(flag, epsilon, M), args.log_file)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr * args.batch_size_per_gpu * args.dist_size / 256.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0)

    if args.criterion == 'smooth':
        write('Using Label Smooth Cross Entropy', args.log_file)
        criterion = LabelSmoothingCrossEntropy()
    elif args.criterion == 'ce':
        write('Using CE Loss', args.log_file)
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

    for ep in range(1, epochs+1):
        classifier.train()
        loss_metric = AvgMetric()
        train_loader.sampler.set_epoch(ep)
        for _it, (images, targets) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            with torch.no_grad():
                features = backbone(images)

            assert (features.size(1) == args.feat_dim) and (len(features.size()) == 2)
            bs = features.size(0)

            with torch.no_grad():
                features = features.unsqueeze(1) + (torch.randn(size=[features.size(0), M, features.size(1)]).cuda() * epsilon * args.train_std.reshape(1,1,-1))
                features = features.reshape(-1, args.feat_dim)
            logits = classifier(features)
            loss = criterion(logits, targets.reshape(-1, 1).repeat(1, M).reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_metric.update(loss.item(), bs)

            write('\rEpoch : {}/{}    Iter : {}/{}    Epsilon : {:.4f}    Lr : {:.6f}    Loss : {:.4f}'.format(ep, epochs, _it+1, len(train_loader), epsilon, scheduler.get_last_lr()[0], loss_metric.show()), end='\n' if _it+1 == len(train_loader) else '', log_file=args.log_file)

        scheduler.step()

    if test_loader is not None:
        write('Start testing when training, epsilon=0 with M=1', args.log_file)
        classifier.eval()
        acc_metric = AvgMetric()
        with torch.no_grad():
            for images, targets, _ in test_loader:
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

                bs = images.size(0)

                with torch.no_grad():
                    features = backbone(images)

                logits = classifier(features)
                acc = accuracy(logits, targets)

                acc_metric.update(acc, bs)

        acc_metric.synchronize()
        write('samples in test loop is {}'.format(acc_metric.n), args.log_file)
        test_acc = acc_metric.show()

        write('Test_acc : {:.4f}'.format(test_acc), args.log_file)

    try:
        return test_acc
    except:
        return

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--shot', default=1, type=int)
    parser.add_argument('--dataset', default='Imagenet', choices=['Imagenet', 'Imanget_1pt', 'Imagenet_10pt'])
    parser.add_argument('--arch', default='deit_large_p7', choices=['deit_large_p7', 'deit_base_p4'])
    parser.add_argument('--pretrain_method', default='MSN', choices=['MSN'])
    parser.add_argument('--round', default=0.95, type=float)

    parser.add_argument('--local_rank', default=0, type=int)

    parser.add_argument('--M', default=200, type=int)
    parser.add_argument('--fc', default='normlinear', choices=['linear', 'normlinear', 'cosine'])
    parser.add_argument('--criterion', default='smooth', choices=['ce', 'smooth'])
    parser.add_argument('--pre_epochs', default=100, type=int)
    parser.add_argument('--cycle_epochs', default=100, type=int)
    parser.add_argument('--final_epochs', default=100, type=int)
    parser.add_argument('--search_lr', default=0.1, type=float)
    parser.add_argument('--train_lr', default=0.5, type=float)
    parser.add_argument('--scale_factor', default=10.0, type=float)
    parser.add_argument('--batch_size_per_gpu', default=32, type=int)
    parser.add_argument('--batch_size_test_per_gpu', default=32, type=int)
    parser.add_argument('--num_workers_per_gpu', default=4, type=int)

    args = parser.parse_args()

    if args.dataset == 'Imagenet':
        args.num_classes = 1000
    elif args.dataset == 'CUB':
        args.num_classes = 200
    else:
        raise NotImplementedError

    if 'deit_small' in args.arch:
        args.feat_dim = 384
    elif 'deit_base' in args.arch:
        args.feat_dim = 768
    elif 'deit_large' in args.arch:
        args.feat_dim = 1024
    else:
        raise NotImplementedError

    if args.pretrain_method == 'MSN':
        if args.arch == 'deit_large_p7':
            args.load_path = 'checkpoint/MSN/vitl7_200ep.pth.tar'
        elif args.arch == 'deit_base_p4':
            args.load_path = 'checkpoint/MSN/vitb4_300ep.pth.tar'
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    #world init
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl')
    args.dist_size = dist.get_world_size()
    setup_for_distributed(args.local_rank == 0)

    exp_dir = os.path.join(PATH, 'checkpoint', args.pretrain_method, args.dataset, args.arch)
    exp_dir_ft = os.path.join(exp_dir, 'CompareWithLinearSVM', 'channel_wise_with_train_aug_search_lr_{}_bs_{}_fc_{}_tao_{}_cri_{}_rd_{}_M_{}_Pre_{}_C_{}_F_{}'.format(args.search_lr, args.batch_size_per_gpu * args.dist_size, args.fc, args.scale_factor, args.criterion, args.round, args.M, args.pre_epochs, args.cycle_epochs, args.final_epochs), '{}shot'.format(args.shot) if args.dataset == 'Imagenet' else args.dataset.split('_')[-1])

    args.exp_dir_ft = exp_dir_ft
    args.exp_dir_ftt = os.path.join(exp_dir_ft, 'naive_bsearch')
    args.log_file = os.path.join(args.exp_dir_ftt, 'train_lr_{:.5f}_naive_bsearch.txt'.format(args.train_lr))

    if args.local_rank == 0:
        if not os.path.exists(args.exp_dir_ftt):
            os.makedirs(args.exp_dir_ftt)

        if os.path.isfile(args.log_file):
            os.remove(args.log_file)

    dist.barrier()
    write(vars(args), args.log_file)

    if args.pretrain_method == 'MSN':
        if args.arch == 'deit_large_p7':
            backbone = deit_large_p7()
        elif args.arch == 'deit_base_p4':
            backbone = deit_base_p4()
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    write(backbone, args.log_file)

    load_dict = torch.load(args.load_path, map_location=torch.device('cpu'))
    if args.pretrain_method == 'MSN':
        load_dict = {k.replace('module.', ''): v for k, v in load_dict['target_encoder'].items() if k.replace('module.', '') in backbone.state_dict()}
    else:
        raise NotImplementedError

    write('len of load dict {}'.format(len(load_dict)), args.log_file)

    msg = str(backbone.load_state_dict(load_dict, strict=False))
    write(msg, args.log_file)
    assert '<All keys matched successfully>' in msg

    backbone.cuda()
    backbone.eval()

    for p in backbone.parameters():
        p.requires_grad = False


    mset = FewShotMetaSet(setname=args.dataset, shot=args.shot)

    test_set = mset.test
    test_loader = torch.utils.data.DataLoader(
        test_set,
        sampler=torch.utils.data.distributed.DistributedSampler(test_set, shuffle=False),
        batch_size=args.batch_size_test_per_gpu,
        num_workers=args.num_workers_per_gpu,
        pin_memory=True)
    write('the number of samples in test set is {}, in test_loader is {}'.format(len(test_set), len(test_loader)), args.log_file)

    accs_naive = []
    accs_bsearch = []

    epsilons = []
    log_path = os.path.join(exp_dir_ft, 'search_log.txt')
    with open(log_path, 'r') as f:
        lines = f.readlines()
        _tmp = lines[-1].strip()
    epsilons.append(float(_tmp.split(' ')[5]))
    epsilons.append(float(_tmp.split(' ')[9]))
    epsilons.append(float(_tmp.split(' ')[13]))

    train_lr = args.train_lr

    for run in range(3):
        write('\n\n', args.log_file)

        train_set = mset.trains[run]
        train_loader = torch.utils.data.DataLoader(
            train_set,
            sampler=torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True),
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers_per_gpu,
            pin_memory=True)
        write('the number of samples in train set is {}, in train_loader is {}'.format(len(train_set), len(train_loader)), args.log_file)

        train_plain_set = mset.trains_plain[run]
        train_plain_loader = torch.utils.data.DataLoader(
            train_plain_set,
            sampler=torch.utils.data.distributed.DistributedSampler(train_plain_set, shuffle=False),
            batch_size=args.batch_size_test_per_gpu,
            num_workers=args.num_workers_per_gpu,
            pin_memory=True)
        write('the number of samples in train set (plain) is {}, in train_loader is {}'.format(len(train_plain_set), len(train_plain_loader)), args.log_file)

        with torch.no_grad():
            all_features = torch.zeros(size=[0, ]).cuda()
            for images, _ in tqdm(train_plain_loader):
                images = images.float().cuda(non_blocking=True)

                features = backbone(images)
                all_features = torch.cat([all_features, features], dim=0)

            write('all_features dim {}'.format(all_features.size()), args.log_file)
            all_features = concat_all_gather(all_features).reshape(-1, args.feat_dim)
            write('samples to compute std is {}'.format(all_features.size(0)), args.log_file)

        #####################################
        args.train_std = torch.std(all_features, dim=0).cuda()
        assert args.train_std.size(0) == args.feat_dim
        write('\n\n', args.log_file)
        #####################################

        epsilon = epsilons[run]
        write('Current epsilon is {:.4}'.format(epsilon), args.log_file)
        write('**********   Final Results   **********', args.log_file)

        naive_classifier = get_classifier(args, parallel=True)
        naive_acc = train_loop(args, naive_classifier, backbone, train_loader, lr=train_lr, epsilon=0.0, M=1, epochs=args.final_epochs, test_loader=test_loader, flag='post training on naive')
        accs_naive.append(naive_acc)

        bsearch_classifier = get_classifier(args, parallel=True)
        bsearch_acc = train_loop(args, bsearch_classifier, backbone, train_loader, lr=train_lr, epsilon=epsilon, M=args.M, epochs=args.final_epochs, test_loader=test_loader, flag='post training on bsearch')
        accs_bsearch.append(bsearch_acc)

        write('***Run {}***    Lr : {:.5f}      Naive : {:.4f}      BSearch : {:.4f}'.format(run, train_lr, naive_acc, bsearch_acc), args.log_file)

    write('***Final Epsilon    {:.4f}    {:.4f}    {:.4f} ***'.format(epsilons[0], epsilons[1], epsilons[2]), args.log_file)
    write('\n', args.log_file)

    acc_naive_mean = np.mean(accs_naive)
    acc_naive_std = np.std(accs_naive)
    acc_bsearch_mean = np.mean(accs_bsearch)
    acc_bsearch_std = np.std(accs_bsearch)

    write('***Final Results***     Lr : {:.5f}      Naive : {:.2f} +- {:.2f}      BSearch : {:.2f} +- {:.2f}'.format(
        train_lr,
        acc_naive_mean * 100., acc_naive_std * 100.,
        acc_bsearch_mean * 100., acc_bsearch_std * 100.,
    ), args.log_file)




if __name__ == '__main__':
    main()