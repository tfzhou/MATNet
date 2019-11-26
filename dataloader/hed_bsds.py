import os
from PIL import Image

from torch.utils import data

class HEDBSDSTrain(data.Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        train_pair = os.path.join(root_dir, 'train_pair.lst')
        with open(train_pair) as f:
            lines = f.readlines()

        self.imagefiles = []
        self.labelfiles = []
        for line in lines:
            splits = line.split()
            self.imagefiles.append(os.path.join(root_dir, splits[0]))
            self.labelfiles.append(os.path.join(root_dir, splits[1]))


    def __len__(self):
        return len(self.imagefiles)


    def __getitem__(self, index):
        imagefile = self.imagefiles[index]
        labelfile = self.labelfiles[index]

        image = Image.open(imagefile).convert('RGB')
        label = Image.open(labelfile).convert('L')

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label


class HEDBSDSTest(data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform

        train_pair = os.path.join(root_dir, 'test.lst')
        with open(train_pair) as f:
            lines = f.readlines()

        self.imagefiles = []
        for line in lines:
            splits = line.split()
            self.imagefiles.append(os.path.join(root_dir, splits[0]))

    def __len__(self):
        return len(self.imagefiles)

    def __getitem__(self, index):
        imagefile = self.imagefiles[index]

        image = Image.open(imagefile).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, imagefile