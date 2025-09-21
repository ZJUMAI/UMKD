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

class DRDataset012(data_utils.Dataset):

    def __init__(self, patch_level, img_root, dataset, transform=None, fold=3):
        self.data_list = []
        # self.items = []
        self.transform = transform
        self.patch_level = patch_level
        if dataset == 'train':
            data_num = [i for i in range(10) if i != fold]
            print("data_num:",data_num)
        elif dataset == 'valid':
            data_num = [fold]
        print("Use the data fold: {}".format(fold))
        for i in data_num:
            #f = open('data_split/dr/fold_{}.csv'.format(i), "r")
            f = open('/data2/dongchunlai/data/ord_reg/DR_dataset/ten_fold//fold_{}.csv'.format(i), "r")
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.data_list.append(row[1:])
        # 提取第二个元素为0、1或2的子列表
        self.data_list = [item for item in self.data_list if item[1] in {'0', '1', '2'}]

        self.label_list = [int(x[-1]) for x in self.data_list] #提出了所有折中的label
        if self.patch_level == 1:
            self.patch_class = self.get_token_class(fold)

        self.label_num = [0, 0, 0, 0, 0]
        for each in self.label_list:
            self.label_num[each] += 1
        print(self.label_num) #[23229, 2199, 4763, 786, 638]

        print(len(self.data_list)) #31615

    def get_token_class(self, fold):
        patch_token_file = '/data2/dongchunlai/Controllable-Image-Generation/patch_token/dr/results_dict_fold{}.pkl'.format(fold)
        print("Use the patch class: " + patch_token_file)
        # patch_token_file = '/data2/dongchunlai/Controllable-Image-Generation/dr_results_dict.pkl'
        with open(patch_token_file, 'rb') as f:
            results_dict = pickle.load(f)

        target_dict = {}
        # 遍历 results_dict 中的每个条目
        for img_path, token_cls in results_dict.items():
            # token_cls 是形状为 (w, h, num_classes) 的张量
            # 计算 token_cls 的 argmax，得到形状为 (w, h) 的目标类别标签
            target = token_cls.argmax(axis=-1)
            # 将目标类别标签存储到 target_dict 中
            target_dict[img_path] = target

        # for img_path, target in target_dict.items():
        #     print(f"Image Path: {img_path}")
        #     print(f"Target Classes:\n{target}")
        #     break  # 仅打印一个示例

        return target_dict
        

    def choose_ref(self, label):
        if label == 0:
            out = 1
        elif label == 4:
            out = 3
        else:
            left_num = self.label_num[label-1]
            right_num = self.label_num[label+1]
            out = label + 1 if random.random() < (left_num/(left_num+right_num)) else label - 1
        return out

    def __getitem__(self, idx):
        item = copy.deepcopy(self.data_list[idx])
        img = item[0]
        label = int(item[1])
        # img_path = '/data3/share/ord_reg/ord_reg/DR_data/train/' + img + '.jpeg'
        img_path = '/data2/dongchunlai/data/ord_reg/DR_dataset/train/' + img + '.jpg'
        # label = int(item[-1])
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label

    #def __getitem__(self, idx):
    #    item = copy.deepcopy(self.data_list[idx])
    #    img = item[0]
    #    label = int(item[1])
    #    # img_path = '/data3/share/ord_reg/ord_reg/DR_data/train/' + img + '.jpeg'
    #    img_path = '/data2/dongchunlai/data/ord_reg/DR_dataset/train/' + img + '.jpg'
    #    # label = int(item[-1])
    #    img = Image.open(img_path).convert('RGB')
    #    if self.patch_level == 1:
    #        patch_classes = self.patch_class[img_path]
    #        if self.transform:
    #            img = self.transform(img)
    #        return img, label, patch_classes
    #    
    #    if self.transform:
    #        img = self.transform(img)
    #    return img, label

    def __len__(self):
        return len(self.data_list)
    
def pretrain_loader(train_set, batch_size, shuffle=True, num_workers=4):
    # transform=transforms.Compose([
    #         transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
    #         transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
    #         transforms.RandomApply([
    #             transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
    #         ], p=0.8),
    #         transforms.RandomGrayscale(p=0.2),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                              std=[0.229, 0.224, 0.225])
    #     ])

    transform1 = transforms.Compose([
            transforms.Resize(size=(256, 256), interpolation=InterpolationMode.BILINEAR), 
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224, scale=(0.95, 1.))
        ])
    transform2 = transforms.Compose([
            TrivialAugmentWideNoShape(),
            transforms.RandomCrop(size=(224, 224)), #includes crop
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    trainset = TwoAugSupervisedDataset(train_set, transform1=transform1, transform2=transform2)
    trainloader_pretraining = torch.utils.data.DataLoader(trainset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                drop_last=True
                                )
    return trainloader_pretraining


class TwoAugSupervisedDataset(torch.utils.data.Dataset):
    r"""Returns two augmentation and no labels."""
    def __init__(self, dataset, transform1, transform2):
        self.dataset = dataset
        # self.classes = dataset.classes
        
        self.transform1 = transform1
        self.transform2 = transform2

    def __getitem__(self, index):
        item = copy.deepcopy(self.dataset.data_list[index])
        img = item[0]
        label = int(item[1])
        # img_path = '/data2/wangjinhong/data/ord_reg/DR_data/train/' + img + '.jpeg'
        img_path = '/data2/chengyi/dataset/ord_reg/DR_dataset/train/' + img + '.jpg'
        # label = int(item[-1])
        img = Image.open(img_path).convert('RGB')

        if self.dataset.transform:
            img_raw = self.dataset.transform(img)

        image = self.transform1(img)

        return img_raw, self.transform2(image), label

    def __len__(self):
        return len(self.dataset)
    
# function copied from https://pytorch.org/vision/stable/_modules/torchvision/transforms/autoaugment.html#TrivialAugmentWide (v0.12) and adapted
class TrivialAugmentWideNoColor(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.5, num_bins), True), 
            "ShearY": (torch.linspace(0.0, 0.5, num_bins), True), 
            "TranslateX": (torch.linspace(0.0, 16.0, num_bins), True), 
            "TranslateY": (torch.linspace(0.0, 16.0, num_bins), True), 
            "Rotate": (torch.linspace(0.0, 60.0, num_bins), True), 
        }

class TrivialAugmentWideNoShapeWithColor(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.5, num_bins), True), 
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

class TrivialAugmentWideNoShape(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.02, num_bins), True), 
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }


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
    # print(tran)
    train_data = DRDataset012(None,None,'train',tran,0)
    # loader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=True, num_workers=4, drop_last=True)
    loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True, drop_last=True)
    for i, (a, b) in enumerate(loader):
        print('i:',i)
        print('###',a.shape, b.shape)
        break
    #for i, (a, b, c, d) in enumerate(loader):
    #   print(i)
    #   print(a.shape, b.shape, c.shape, d.shape)
    #    print(c, d)
    #    break


#if __name__ == '__main__':
#    for fold in range(10):
#        MyDataset(None,None,'valid',fold=fold)