import torch
import torch.utils.data as data_utils
import numpy as np
import os
from PIL import Image, ImageDraw
from PIL import Image
import copy
import pandas as pd
import csv
import random
from typing import Tuple, Dict
from torch import Tensor
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import pickle

class SICAPv2Dataset(data_utils.Dataset):

    def __init__(self, patch_level, img_root, dataset, transform=None, fold=3, balanced=False):
        self.data_list = []
        # self.items = []
        self.transform = transform
        self.patch_level = patch_level
        self.balanced = balanced
        # if dataset == 'train':
        #     data_num = [i for i in range(10) if i != fold]
        # elif dataset == 'valid':
        #     data_num = [fold]
#########训练和测试集转换为8：2##########
        if dataset == 'train':
            data_num = [i for i in range(10) if i != fold and i != (fold + 1) % 10]
        elif dataset == 'valid':
            data_num = [fold, (fold + 1) % 10]
        print("Use the data fold: {}".format(data_num))
        for i in data_num:
            if self.balanced:
                f = open('/data3/tongshuo/dataset/image/SICAPv2/ten_fold_balanced/fold_{}.csv'.format(i), "r")
            else:
                f = open('/data3/tongshuo/dataset/image/SICAPv2/ten_fold/fold_{}.csv'.format(i), "r")
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.data_list.append(row[1:])

        self.label_list = [int(x[-1]) for x in self.data_list] #提出了所有折中的label
        # 提取出0, 1, 2
        #self.label_list = [x for x in self.label_list if x in {0, 1, 2}]

        if self.patch_level == 1:
            self.patch_class = self.get_token_class(fold)

        self.label_num = [0, 0, 0, 0]
        for each in self.label_list:
            self.label_num[each] += 1
        print(self.label_num) #[23229, 2199, 4763, 786, 638]
        print(len(self.data_list)) #31615

    def __getitem__(self, idx):
        item = copy.deepcopy(self.data_list[idx])
        # print(item)
        img = item[0]
        label = int(item[1])
        img_path = '/data3/tongshuo/dataset/image/SICAPv2/images/' + img
        # label = int(item[-1])
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    tran = transforms.Compose([
            transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    train_data = SICAPv2Dataset(None,None,'train',tran,0)
    # loader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=True, num_workers=4, drop_last=True)
    loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True, drop_last=True)
    for i, (a, b) in enumerate(loader):
        print('i:',i)
        print('###',a.shape, b.shape)
        break