import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pprint
import math
import bisect
import logging
import sys
import os
import colorlog
import copy
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.data import Sampler
# from core.modules import NormDistBase

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

class AvgMetric:
    def __init__(self):
        self.reset()

    def reset(self):
        self.v = 0.0
        self.n = 0.0

    def update(self, v, n):
        self.v += v * n
        self.n += n

    def show(self):
        return self.v / self.n

    @torch.no_grad()
    def synchronize(self):
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.v, self.n], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.v = float(t[0])
        self.n = float(t[1])


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss_func with label smoothing.
    """
    def __init__(self, smoothing=0.1, reduction='mean'):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
        self.reduction = reduction

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss



@torch.no_grad()
def accuracy(y_pred, y):
    return float(torch.eq(y_pred.argmax(dim=-1), y).sum().item()) / float(y_pred.shape[0])


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.stack(tensors_gather, dim=0)
    return output

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)


def write(print_obj, log_file=None, end='\n'):
    print(print_obj, end=end)
    if log_file is not None:
        with open(log_file, 'a') as f:
            print(print_obj, end=end, file=f)

log_colors_config = {
    'DEBUG': 'cyan',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red',
}

def setup_logger(name, save_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch_formatter = colorlog.ColoredFormatter("%(asctime)s %(name)s %(levelname)s: %(log_color)s %(message)s", log_colors=log_colors_config)
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "log.txt"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)

    return logger



def train_loop(args, classifier, train_loader, epsilon, epochs, lr_decay=True, test_loader=None):
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-5)

    if args.criterion == 'smooth':
        criterion = LabelSmoothingCrossEntropy()
    elif args.criterion == 'ce':
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

    best_test_acc = 0.0
    for ep in range(1, epochs+1):
        classifier.train()
        loss_metric = AvgMetric()
        for _it, (features, targets, _) in enumerate(train_loader):
            features = features.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            bs = features.size(0)

            delta = torch.randn_like(features).cuda() * epsilon
            logits = classifier(features + delta)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_metric.update(loss.item(), bs)

            write('\rEpoch : {}/{}    Iter : {}/{}    Epsilon : {:.4f}    Lr : {:.6f}    Loss : {:.4f}'.format(ep, epochs, _it+1, len(train_loader), epsilon, scheduler.get_last_lr()[0], loss_metric.show()), end='\n' if _it+1 == len(train_loader) else '', log_file=args.log_file)

        if lr_decay:
            scheduler.step()

        if (test_loader is not None) and (ep % 10 == 0):

            classifier.eval()
            acc_metric = AvgMetric()

            with torch.no_grad():
                for features, targets, _ in test_loader:
                    features = features.cuda(non_blocking=True)
                    targets = targets.cuda(non_blocking=True)

                    bs = features.size(0)

                    delta = torch.randn_like(features).cuda() * epsilon
                    logits = classifier(features + delta)
                    acc = accuracy(logits, targets)

                    acc_metric.update(acc, bs)

            write('samples in test loop is {}'.format(acc_metric.n), args.log_file)
            test_acc = acc_metric.show()
            if test_acc > best_test_acc:
                best_test_acc = test_acc

                # if args.local_rank == 0:
                #     save_dir = os.path.join(args.exp_dir_ft, 'weights')
                #
                #     if not os.path.exists(save_dir):
                #         os.makedirs(save_dir)
                #     torch.save(model.state_dict(), os.path.join(save_dir, 'best.pth'))

            write('Test_acc : {:.4f}'.format(test_acc), args.log_file)

@torch.no_grad()
def test_loop(args, classifier, loader, epsilon):
    classifier.eval()
    acc_metric = AvgMetric()
    for features, targets, _ in loader:
        features = features.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        bs = features.size(0)

        delta = torch.randn_like(features).cuda() * epsilon
        logits = classifier(features + delta)
        acc = accuracy(logits, targets)

        acc_metric.update(acc, bs)

    write('samples in test loop is {}'.format(acc_metric.n), args.log_file)
    test_acc = acc_metric.show()

    # write('Test_acc : {:.4f}'.format(test_acc), args.log_file)

    return test_acc


def max_acc_check(lines, naive_acc, bsearch_acc):
    naive_accs = []
    bsearch_accs = []
    for line in lines:
        if ('Acc' in line) and ('Naive' in line):
            gradient = line.strip().split(' ')[-1].split('+-')
            naive_accs.append((float(gradient[0]), float(gradient[1])))

        if ('Acc' in line) and ('BSearch' in line):
            gradient = line.strip().split(' ')[-1].split('+-')
            bsearch_accs.append((float(gradient[0]), float(gradient[1])))

    max_naive = max(naive_accs, key=lambda x: (float(x[0]), -float(x[1])))
    max_bsearch = max(bsearch_accs, key=lambda x: (float(x[0]), -float(x[1])))

    assert abs(max_naive[0] - float(naive_acc.split(' ')[0])) < 1e-6
    assert abs(max_bsearch[0] - float(bsearch_acc.split(' ')[0])) < 1e-6

    if len(naive_acc.split(' ')) > 1:
        assert abs(max_naive[1] - float(naive_acc.split(' ')[2])) < 1e-6
    if len(bsearch_acc.split(' ')) > 1:
        assert abs(max_bsearch[1] - float(bsearch_acc.split(' ')[2])) < 1e-6