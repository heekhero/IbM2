import os

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from datasets.aux_dataset import FolderDataset


class MetaSet:
    def __init__(self, data_path):

        test_split = 'val'

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

        self.train = FolderDataset(root=os.path.join(data_path, 'train'), transform=train_preprocess)
        self.train_plain = FolderDataset(root=os.path.join(data_path, 'train'), transform=test_preprocess)
        self.test = FolderDataset(root=os.path.join(data_path, test_split), transform=test_preprocess)
