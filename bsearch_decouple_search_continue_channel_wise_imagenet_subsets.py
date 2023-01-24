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

def train_loop(args, classifier, train_loader, lr, epsilon, M, epochs, test_loader=None, test_M=1, test_epsilon=0.0, flag='training'):
    write('Starting {} from epsilon={:.4f} with M={}'.format(flag, epsilon, M), args.log_file)
    if args.opt == 'adam':
        write('Using torch.optim.Adam(classifier.parameters(), lr=lr * args.batch_size_per_gpu / 256.) as optimizer', args.log_file)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr * args.batch_size_per_gpu_search / 256.)
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
        write('Start testing when training, epsilon={} with M={}'.format(test_epsilon, test_M), args.log_file)
        classifier.eval()
        acc_metric = AvgMetric()
        with torch.no_grad():
            for features, targets, _ in test_loader:
                features = features.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

                bs = features.size(0)

                features = features.unsqueeze(1) + (torch.randn(size=[features.size(0), test_M, features.size(1)]).cuda() * test_epsilon * args.train_std.reshape(1, 1, -1))
                features = features.reshape(-1, args.feat_dim)
                logits = classifier(features)
                acc = accuracy(logits, targets.reshape(-1, 1).repeat(1, test_M).reshape(-1))

                acc_metric.update(acc, bs)

        acc_metric.synchronize()
        write('samples in test loop is {}'.format(acc_metric.n), args.log_file)
        test_acc = acc_metric.show()

        write('Test_acc : {:.4f}'.format(test_acc), args.log_file)

        return test_acc

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pt', default=1, type=int)
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

    args.dataset = 'Imagenet_{}pt'.format(args.pt)
    args.num_classes = 1000

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
    exp_dir_ft = os.path.join(exp_dir, 'BSearch_recycle_adaptive_th_decouple_search_continue_channel_wise_new_no_cyan', 'pre_norm_True_opt_{}_search_lr_{}_search_bs_{}_train_bs_{}_fc_{}_tao_{}_cri_{}_rd_{}_right_{}_M_{}_Pre_{}_C_{}_F_{}_seed_{}'.format(args.opt, args.search_lr, args.batch_size_per_gpu_search, args.batch_size_per_gpu_train, args.fc, args.scale_factor, args.criterion, args.round, args.right, args.M, args.pre_epochs, args.cycle_epochs, args.final_epochs, args.seed), '{}pt'.format(args.pt))

    args.exp_dir_ft = exp_dir_ft
    args.log_file = os.path.join(exp_dir_ft, 'search_log.txt')

    if not os.path.exists(exp_dir_ft):
        os.makedirs(exp_dir_ft)

    if os.path.isfile(args.log_file):
        os.remove(args.log_file)

    write(vars(args), args.log_file)


    write('\n\n', args.log_file)

    args.train_path = os.path.join(DATA3_ROOT_PATH, 'checkpoint', args.pretrain_method, args.dataset, args.arch, 'features', 'train_{}pt_normalized_no_cyan.pth'.format(args.pt))
    train_set = FeatureDataset(args.train_path)
    write('the number of samples in train set is {}'.format(len(train_set)), args.log_file)

    #####################################
    args.train_std = torch.std(train_set.data, dim=0).cuda()
    assert args.train_std.size(0) == args.feat_dim
    write('\n\n', args.log_file)
    #####################################

    write('Std in train(plain) set is {}'.format(args.train_std), args.log_file)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=args.batch_size_per_gpu_search, num_workers=args.num_workers_per_gpu, pin_memory=True)

    init_classifier = get_classifier(args, parallel=False)
    write(init_classifier, args.log_file)

    write('Searching layers is feature layer', args.log_file)

    write('**********   Baseline   **********', args.log_file)
    th = train_loop(args, init_classifier, train_loader, lr=args.search_lr, epsilon=0.0, M=1, epochs=args.pre_epochs, test_loader=train_loader, test_M=1, test_epsilon=0.0, flag='training before search (to get th)')

    write('Naive test_acc in train set with no epsilon is : {:.2f}'.format(th), args.log_file)
    th = min(args.round, th)


    write('\n', args.log_file)
    write('\n', args.log_file)

    write('**********   Searching Stage   **********', args.log_file)
    write('Threshold : {:.2f}'.format(th), args.log_file)

    left_bound = 0.0
    right_bound = args.right
    epsilon = right_bound / 2

    search_classifier = get_classifier(args, parallel=False)
    while True:
        train_acc_now = train_loop(args, search_classifier, train_loader, lr=args.search_lr, epsilon=epsilon, M=args.M, epochs=args.cycle_epochs, test_loader=train_loader, test_M=args.M, test_epsilon=epsilon, flag='searching')

        if train_acc_now > th:
            left_bound = epsilon  # harder
        else:
            right_bound = epsilon  # easier

        epsilon = (left_bound + right_bound) / 2.0

        if right_bound - left_bound < 0.05:
            break

    write('Epsilon after search is {:.4f}'.format(epsilon), args.log_file)


    write('***Final Epsilon    {:.4f}    ***'.format(epsilon), args.log_file)






if __name__ == '__main__':
    main()