import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

from torch.nn.utils.weight_norm import WeightNorm

class CosineClassifier(nn.Module):
    def __init__(self, indim, outdim, scale_factor):
        super(CosineClassifier, self).__init__()
        self.weight = nn.Parameter(torch.randn(size=[outdim, indim]).float())
        self.weight.requires_grad = True

        self.scale_factor = scale_factor


    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-5)

        weight_norm = torch.norm(self.weight, p=2, dim=1).unsqueeze(1).expand_as(self.weight)
        weight_normalized = self.weight.div(weight_norm + 1e-5)

        cos_dist = F.linear(x_normalized, weight_normalized)
        scores = self.scale_factor * cos_dist

        return scores

class NormalizedLinear(nn.Module):
    def __init__(self, indim, outdim, scale_factor):
        super(NormalizedLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias=False)
        WeightNorm.apply(self.L, 'weight', dim=0)

        self.scale_factor = scale_factor


    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-5)
        norm_dist = self.L(x_normalized)
        scores = self.scale_factor * norm_dist

        return scores

def get_classifier(args, parallel=False):

    if args.fc == 'linear':
        classifier = nn.Linear(args.feat_dim, args.num_classes)
        classifier.weight.data.normal_(mean=0.0, std=0.01)
        classifier.bias.data.zero_()
    elif args.fc == 'normlinear':
        classifier = NormalizedLinear(args.feat_dim, args.num_classes, args.scale_factor)
    elif args.fc == 'cosine':
        classifier = CosineClassifier(args.feat_dim, args.num_classes, args.scale_factor)
    else:
        raise NotImplementedError


    classifier.cuda()
    if parallel:
        classifier = nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
        classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[args.local_rank], output_device=args.local_rank)

    return classifier

if __name__ == '__main__':
    net = CosineClassifier(512, 5, 10)
    x = torch.randn(size=[16, 512])

    y = net(x)
    loss = y.sum()
    loss.backward()
    print()