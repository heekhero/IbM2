from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os
import random

from datasets.aux_dataset import FolderDataset
from config import DATA_PATH, SPLIT_PATH, HOST_NAME

from collections import defaultdict
from torchvision.transforms.functional import InterpolationMode

class MetaSet:
    def __init__(self, dataset):

        if 'ubuntu' in HOST_NAME:
            prefix = 'root'
        else:
            prefix = 'opt'

        if dataset == 'Imagenet':
            data_dir = '/{}/Dataset/ImageNet'.format(prefix)
            test_split = 'val'
        elif dataset == 'Imagenet_1pt':
            data_dir = os.path.join(DATA_PATH, 'ImageNet_subsets', '1pt')
            test_split = 'val'
        elif dataset == 'Imagenet_10pt':
            data_dir = os.path.join(DATA_PATH, 'ImageNet_subsets', '10pt')
            test_split = 'val'
        elif dataset == 'CUB':
            data_dir = os.path.join(DATA_PATH, 'CUB')
            test_split = 'test'
        else:
            raise NotImplementedError

        self.dataset_dir = data_dir


        train_preprocess = transforms.Compose([
            transforms.RandomResizedCrop(size=224, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


        test_preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.train = FolderDataset(root=os.path.join(self.dataset_dir, 'train'), transform=train_preprocess)
        self.train_plain = FolderDataset(root=os.path.join(self.dataset_dir, 'train'), transform=test_preprocess)
        self.test = FolderDataset(root=os.path.join(self.dataset_dir, test_split), transform=test_preprocess)
