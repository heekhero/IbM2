import torch
import torch.nn as nn

from torch.nn.utils.weight_norm import WeightNorm

class NormalizedLinear(nn.Module):
    def __init__(self, indim, outdim, scale_factor=10.0):
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

def get_classifier(args):

    classifier = NormalizedLinear(args.feat_dim, args.num_classes)
    classifier.cuda()

    return classifier