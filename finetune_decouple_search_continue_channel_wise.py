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

from tqdm import tqdm
from collections import defaultdict, OrderedDict
from config import PATH, DATA3_ROOT_PATH
from model.classifier import get_classifier
from datasets.aux_dataset import FeatureDataset
from utils import accuracy, AvgMetric, write, setup_for_distributed, LabelSmoothingCrossEntropy

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.benchmark = True

def train_loop(args, classifier, train_loader, lr, epsilon, M, epochs, test_loader=None, flag='training'):
    write('Starting {} from epsilon={:.4f} with M={}'.format(flag, epsilon, M), args.log_file)
    if args.opt == 'adam':
        write('Using torch.optim.Adam(classifier.parameters(), lr=lr * args.batch_size_per_gpu / 256.) as optimizer', args.log_file)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr * args.batch_size_per_gpu_train / 256.)
    else:
        raise NotImplementedError

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
        for _it, (features, targets, _) in enumerate(train_loader):
            features = features.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

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
            for features, targets, _ in test_loader:
                features = features.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

                bs = features.size(0)

                logits = classifier(features)
                acc = accuracy(logits, targets)

                acc_metric.update(acc, bs)

        write('samples in test loop is {}'.format(acc_metric.n), args.log_file)
        test_acc = acc_metric.show()

        write('Test_acc : {:.4f}'.format(test_acc), args.log_file)

        return test_acc

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--shot', default=1, type=int)
    parser.add_argument('--dataset', default='CUB', choices=['Imagenet', 'CUB'])
    parser.add_argument('--arch', default='deit_small_p16', choices=['deit_small_p16', 'deit_large_p7', 'deit_base_p4', 'deit_small_p8', 'resnet50'])
    parser.add_argument('--pretrain_method', default='MoCov3', choices=['DINO', 'MSN', 'MoCov3', 'iBOT', 'SimCLR', 'BYOL', 'SwAV'])
    parser.add_argument('--round', default=0.95, type=float)


    parser.add_argument('--opt', default='adam')
    parser.add_argument('--M', default=200, type=int)
    parser.add_argument('--fc', default='normlinear', choices=['linear', 'normlinear', 'cosine'])
    parser.add_argument('--criterion', default='smooth', choices=['ce', 'smooth'])
    parser.add_argument('--pre_epochs', default=60, type=int)
    parser.add_argument('--cycle_epochs', default=20, type=int)
    parser.add_argument('--final_epochs', default=60, type=int)
    parser.add_argument('--search_lr', default=1.0, type=float)
    parser.add_argument('--train_lr', default=0.5, type=float)
    parser.add_argument('--scale_factor', default=10.0, type=float)
    parser.add_argument('--right', default=10.0, type=float)
    parser.add_argument('--batch_size_per_gpu_search', default=256, type=int)
    parser.add_argument('--batch_size_per_gpu_train', default=256, type=int)
    parser.add_argument('--batch_size_test_per_gpu', default=256, type=int)
    parser.add_argument('--num_workers_per_gpu', default=0, type=int)
    parser.add_argument('--seed', default=0, type=int)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

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
    elif 'resnet50' in args.arch:
        args.feat_dim = 2048
    else:
        raise NotImplementedError

    exp_dir = os.path.join(PATH, 'checkpoint', args.pretrain_method, args.dataset, args.arch)
    exp_dir_ft = os.path.join(exp_dir, 'BSearch_recycle_adaptive_th_decouple_search_continue_channel_wise_new_no_cyan', 'pre_norm_True_opt_{}_search_lr_{}_search_bs_{}_train_bs_{}_fc_{}_tao_{}_cri_{}_rd_{}_right_{}_M_{}_Pre_{}_C_{}_F_{}_seed_{}'.format(args.opt, args.search_lr, args.batch_size_per_gpu_search, args.batch_size_per_gpu_train, args.fc, args.scale_factor, args.criterion, args.round, args.right, args.M, args.pre_epochs, args.cycle_epochs, args.final_epochs, args.seed), '{}shot'.format(args.shot))

    args.exp_dir_ft = exp_dir_ft
    args.exp_dir_ftt = os.path.join(exp_dir_ft, 'naive_bsearch')
    args.log_file = os.path.join(args.exp_dir_ftt, 'train_lr_{:.5f}_naive_bsearch.txt'.format(args.train_lr))

    if not os.path.exists(args.exp_dir_ftt):
        os.makedirs(args.exp_dir_ftt)

    if os.path.isfile(args.log_file):
        os.remove(args.log_file)

    write(vars(args), args.log_file)

    args.test_path = os.path.join(DATA3_ROOT_PATH, 'checkpoint', args.pretrain_method, args.dataset, args.arch, 'features', 'test_normalized_no_cyan.pth')
    test_set = FeatureDataset(args.test_path)
    write('the number of samples in test set is {}'.format(len(test_set)), args.log_file)

    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=args.batch_size_test_per_gpu, num_workers=args.num_workers_per_gpu, pin_memory=True)

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

        args.train_path = os.path.join(DATA3_ROOT_PATH, 'checkpoint', args.pretrain_method, args.dataset, args.arch, 'features', 'train_{}shot_{}_normalized_no_cyan.pth'.format(args.shot, run))
        train_set = FeatureDataset(args.train_path)
        write('Run : {}    the number of samples in train set is {}'.format(run, len(train_set)), args.log_file)

        #####################################
        args.train_std = torch.std(train_set.data, dim=0).cuda()
        assert args.train_std.size(0) == args.feat_dim
        write('\n\n', args.log_file)
        #####################################

        write('Run : {}    Std in train(plain) set is {}'.format(run, args.train_std), args.log_file)
        train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=args.batch_size_per_gpu_train, num_workers=args.num_workers_per_gpu, pin_memory=True)

        epsilon = epsilons[run]
        write('Current epsilon is {:.4}'.format(epsilon), args.log_file)
        write('**********   Final Results   **********', args.log_file)



        naive_classifier = get_classifier(args, parallel=False)
        naive_acc = train_loop(args, naive_classifier, train_loader, lr=train_lr, epsilon=0.0, M=1, epochs=args.final_epochs, test_loader=test_loader, flag='post training on naive')
        accs_naive.append(naive_acc)

        bsearch_classifier = get_classifier(args, parallel=False)
        bsearch_acc = train_loop(args, bsearch_classifier, train_loader, lr=train_lr, epsilon=epsilon, M=args.M, epochs=args.final_epochs, test_loader=test_loader, flag='post training on bsearch')
        accs_bsearch.append(bsearch_acc)


        write('***Run {}***    Lr : {:.5f}      Naive : {:.4f}      BSearch : {:.4f}'.format(run, train_lr, naive_acc, bsearch_acc), args.log_file)

    write('***Final Epsilon    {:.4f}    {:.4f}    {:.4f} ***'.format(epsilons[0], epsilons[1], epsilons[2]), args.log_file)
    write('\n', args.log_file)

    assert len(accs_naive) == 3
    assert len(accs_bsearch) == 3

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