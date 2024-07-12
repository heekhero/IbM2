import argparse
import os
import random

import numpy as np
import torch.backends.cudnn
import torch.optim
import torch.utils.data

from config import PATH
from datasets.aux_dataset import FeatureDataset
from model.classifier import get_classifier
from utils import accuracy, AvgMetric, write, LabelSmoothingCrossEntropy

torch.backends.cudnn.benchmark = True

def train_loop(args, classifier, train_loader, lr, epsilon, M, epochs, test_loader=None, test_M=1, test_epsilon=0.0, flag='training'):
    write('Starting {} from epsilon={:.4f} with M={}'.format(flag, epsilon, M), args.log_file)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr * args.batch_size_per_gpu / 256.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0)
    criterion = LabelSmoothingCrossEntropy()

    for ep in range(1, epochs+1):
        classifier.train()
        loss_metric = AvgMetric()
        for _it, (features, targets, _) in enumerate(train_loader):
            features = features.float().cuda(non_blocking=True)
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
                features = features.float().cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

                bs = features.size(0)

                features = features.unsqueeze(1) + (torch.randn(size=[features.size(0), test_M, features.size(1)]).cuda() * test_epsilon * args.train_std.reshape(1, 1, -1))
                features = features.reshape(-1, args.feat_dim)
                logits = classifier(features)
                acc = accuracy(logits, targets.reshape(-1, 1).repeat(1, test_M).reshape(-1))

                acc_metric.update(acc, bs)

        write('samples in test loop is {}'.format(acc_metric.n), args.log_file)
        test_acc = acc_metric.show()

        write('Test_acc : {:.4f}'.format(test_acc), args.log_file)

        return test_acc

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--shot', default=1, type=int)
    parser.add_argument('--dataset', default='Imagenet', choices=['Imagenet', 'CUB'])
    parser.add_argument('--arch', default='deit_small_p16', choices=['deit_small_p16', 'deit_large_p7', 'deit_base_p4', 'resnet50'])
    parser.add_argument('--pretrain_method', default='DINO', choices=['DINO', 'MSN', 'MoCov3', 'SimCLR', 'BYOL', 'CLIP', 'DenseCL'])
    parser.add_argument('--round', default=0.9, type=float)

    parser.add_argument('--M', default=200, type=int)
    parser.add_argument('--pre_epochs', default=100, type=int)
    parser.add_argument('--cycle_epochs', default=20, type=int)
    parser.add_argument('--final_epochs', default=100, type=int)
    parser.add_argument('--search_lr', default=1.0, type=float)
    parser.add_argument('--right', default=10.0, type=float)
    parser.add_argument('--batch_size_per_gpu', default=256, type=int)
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
    exp_dir_ft = os.path.join(exp_dir, 'IbM2', 'search_lr_{}_bs_{}_rd_{}_right_{}_M_{}_Pre_{}_C_{}_F_{}_seed_{}'.format(args.search_lr, args.batch_size_per_gpu, args.round, args.right, args.M, args.pre_epochs, args.cycle_epochs, args.final_epochs, args.seed), '{}shot'.format(args.shot))

    args.exp_dir_ft = exp_dir_ft
    args.log_file = os.path.join(exp_dir_ft, 'search_log.txt')

    if not os.path.exists(exp_dir_ft):
        os.makedirs(exp_dir_ft)

    if os.path.isfile(args.log_file):
        os.remove(args.log_file)

    write(vars(args), args.log_file)

    epsilons = []

    for run in range(3):

        write('\n\n', args.log_file)

        args.train_path = os.path.join(PATH, 'checkpoint', args.pretrain_method, args.dataset, args.arch, 'features', 'train_{}shot_{}.pth'.format(args.shot, run))
        train_set = FeatureDataset(args.train_path)
        write('Run : {}    the number of samples in train set is {}'.format(run, len(train_set)), args.log_file)

        #####################################
        args.train_std = torch.std(train_set.data, dim=0).cuda()
        assert args.train_std.size(0) == args.feat_dim
        write('\n\n', args.log_file)
        #####################################

        write('Run : {}    Std in train(plain) set is {}'.format(run, args.train_std), args.log_file)
        train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=args.batch_size_per_gpu, num_workers=args.num_workers_per_gpu, pin_memory=True)

        init_classifier = get_classifier(args)
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

        search_classifier = get_classifier(args)
        while True:
            train_acc_now = train_loop(args, search_classifier, train_loader, lr=args.search_lr, epsilon=epsilon, M=args.M, epochs=args.cycle_epochs, test_loader=train_loader, test_M=args.M, test_epsilon=epsilon, flag='searching')

            if train_acc_now > th:
                left_bound = epsilon  # harder
            else:
                right_bound = epsilon  # easier

            epsilon = (left_bound + right_bound) / 2.0

            if right_bound - left_bound < 0.05:
                break

        epsilons.append(epsilon)
        write('Epsilon after search is {:.4f}'.format(epsilon), args.log_file)


    write('***Final Epsilon    {:.4f}    {:.4f}    {:.4f} ***'.format(epsilons[0], epsilons[1], epsilons[2]), args.log_file)



if __name__ == '__main__':
    main()