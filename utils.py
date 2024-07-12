import pprint

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

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

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)


def write(print_obj, log_file=None, end='\n'):
    print(print_obj, end=end)
    if log_file is not None:
        with open(log_file, 'a') as f:
            print(print_obj, end=end, file=f)