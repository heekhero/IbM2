from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from torchvision.datasets.folder import ImageFolder

import pickle
import torch


class FolderDataset(ImageFolder):
    def __init__(self, root, transform):
        super(FolderDataset, self).__init__(root=root, transform=transform)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
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

    # def parse(self, feature_dict):
    #     data = []
    #     label = []
    #     for cls_id, features in feature_dict.items():
    #         for feat in features:
    #             data.append(feat)
    #             label.append(cls_id)
    #     return data, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.labels[item], item

    # def few_shotlize(self):
    #     split_by_label_dict = defaultdict(list)
    #     for i in range(len(self.data)):
    #         split_by_label_dict[self.labels[i].item()].append(self.data[i])
    #     data = []
    #     labels = []
    #
    #     for label, items in split_by_label_dict.items():
    #         data = data + random.sample(items, self.num_shot)
    #         labels = labels + [label for i in range(self.num_shot)]
    #
    #     self.data = torch.stack(data, dim=0)
    #     self.labels = labels