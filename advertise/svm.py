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
import init_path

from tqdm import tqdm
from collections import defaultdict, OrderedDict
from config import PATH
from model.classifier import get_classifier
from datasets.aux_dataset import FeatureDataset
from utils import accuracy, AvgMetric, write, setup_for_distributed, LabelSmoothingCrossEntropy
from sklearn.svm import SVC

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['OMP_NUM_THREADS'] = '1'
# torch.backends.cudnn.benchmark = True


def accuracy_in_np(y_pred, y):
    return float(np.sum(y_pred == y)) / float(y_pred.shape[0])

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--shot', default=1, type=int)
    parser.add_argument('--arch', default='deit_large_p7', choices=['deit_large_p7', 'deit_base_p4'])
    parser.add_argument('--pretrain_method', default='MSN', choices=['MSN'])
    parser.add_argument('--dataset', default='Imagenet', choices=['Imagenet', 'Imagenet_1pt', 'Imagenet_10pt'])

    parser.add_argument('--aug_100times', default=True, action='store_true')

    args = parser.parse_args()


    args.num_classes = 1000

    if 'deit_base' in args.arch:
        args.feat_dim = 768
    elif 'deit_large' in args.arch:
        args.feat_dim = 1024
    else:
        raise NotImplementedError

    args.exp_dir = os.path.join(PATH, 'checkpoint', args.pretrain_method, args.dataset, args.arch)
    args.exp_dir_ft = os.path.join(args.exp_dir, 'CompareWithLinearSVM', 'aug100times_svm' if args.aug_100times else 'naive_svm', '{}shot'.format(args.shot) if args.dataset == 'Imagenet' else args.dataset.split('_')[-1])

    args.log_file = os.path.join(args.exp_dir_ft, 'results_aug_100times_{}.txt'.format(args.aug_100times))

    if not os.path.exists(args.exp_dir_ft):
        os.makedirs(args.exp_dir_ft)

    if os.path.isfile(args.log_file):
        os.remove(args.log_file)
    write(vars(args), args.log_file)

    args.test_path = os.path.join(PATH, 'checkpoint', args.pretrain_method, args.dataset, args.arch, 'features', 'test.pth')
    test_set = FeatureDataset(args.test_path)
    write('the number of samples in test set is {}'.format(len(test_set)), args.log_file)

    test_X, test_Y = np.array(test_set.data), np.array(test_set.labels)

    accs = []

    for run in range(3):

        write('\n\n', args.log_file)

        train_file_name = 'train_{}shot_{}'.format(args.shot, run) + ('_aug100times.pth' if args.aug_100times else '.pth')
        args.train_path = os.path.join(PATH, 'checkpoint', args.pretrain_method, args.dataset, args.arch, 'features', train_file_name)
        train_set = FeatureDataset(args.train_path)
        write('Run : {}    the number of samples in train set is {}'.format(run, len(train_set)), args.log_file)

        train_X, train_Y = np.array(train_set.data), np.array(train_set.labels)

        model = SVC(kernel='linear', verbose=True)
        model.fit(train_X, train_Y)
        preds = model.predict(test_X)

        acc = accuracy_in_np(preds, test_Y)
        accs.append(acc)

        write('***Run {}***      Test_acc : {:.4f}'.format(run, acc), args.log_file)

    write('\n', args.log_file)

    acc_mean = np.mean(accs)
    acc_std = np.std(accs)

    write('***Final Results***     Test_acc : {:.2f} +- {:.2f}'.format(
        acc_mean * 100., acc_std * 100.,
    ), args.log_file)



if __name__ == '__main__':
    main()