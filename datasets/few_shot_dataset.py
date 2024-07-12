from torch.utils.data import Dataset
from torchvision import transforms
import os

from PIL import Image
from datasets.aux_dataset import FolderDataset
from config import SPLIT_PATH

from torchvision.transforms.functional import InterpolationMode

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

class FewShotMetaSet:
    def __init__(self, data_path, setname, shot):

        if setname == 'Imagenet':
            test_map = 'val'
        elif setname == 'CUB':
            test_map = 'test'
        else:
            raise NotImplementedError

        self.test = FolderDataset(root=os.path.join(data_path, test_map), transform=test_preprocess)
        self.cls2idx = self.test.class_to_idx

        self.trains = [FewShotSet(data_path=os.path.join(data_path, 'train'), split='{}/{}/{}shot/{}.txt'.format(SPLIT_PATH, setname, shot, i), cls2idx=self.cls2idx, transform=train_preprocess) for i in range(3)]
        self.trains_plain = [FewShotSet(data_path=os.path.join(data_path, 'train'), split='{}/{}/{}shot/{}.txt'.format(SPLIT_PATH, setname, shot, i), cls2idx=self.cls2idx, transform=test_preprocess) for i in range(3)]


class FewShotSet(Dataset):
    def __init__(self, data_path, split, cls2idx, transform):
        self.data_path = data_path
        self.split = split
        self.cls2idx = cls2idx
        self.transform = transform

        print('using {}'.format(transform))

        self.parse()

    def parse(self):
        self.img_paths = []
        self.targets = []
        with open(self.split, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                _line = line.split(' ')
                cls_name = _line[0]
                file_name = _line[1]
                file_path = os.path.join(self.data_path, cls_name, file_name)
                self.img_paths.append(file_path)
                self.targets.append(self.cls2idx[cls_name])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        path = self.img_paths[item]
        img = Image.open(path).convert('RGB')
        tensor = self.transform(img)
        target = self.targets[item]

        return tensor, target



if __name__ == '__main__':
    # FewShotSet(data_path='/opt/Dataset/ImageNet/train', split='few_shot_split/Imagenet/1shot/0.txt')
    dataset = FewShotMetaSet('Imagenet', 1)
    print()