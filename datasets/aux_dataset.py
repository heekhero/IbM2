import pickle

import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import ImageFolder


class FolderDataset(ImageFolder):
    def __init__(self, root, transform):
        super(FolderDataset, self).__init__(root=root, transform=transform)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index


class FeatureDataset(Dataset):
    def __init__(self, feature_path):
        with open(feature_path, 'rb') as f:
            feature_dict = pickle.load(f)
        self.data = feature_dict['features']
        self.labels = feature_dict['targets']
        self.classes = torch.unique(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.labels[item], item
