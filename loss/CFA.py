import argparse
import ssl
from abc import ABC

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import random
import warnings
from tqdm import tqdm
from typing import List, Tuple 
from os import mkdir, remove
from os.path import exists
import matplotlib.pyplot as plt

class CommonFeatureLearningLoss(torch.nn.Module):
    def __init__(self, beta: float = 1.0):
        super(CommonFeatureLearningLoss, self).__init__()
        self.beta = beta

    def forward(self, hs: torch.Tensor, ht: torch.Tensor, ft_: torch.Tensor, ft: torch.Tensor) -> torch.Tensor:
        kl_loss = 0.0
        mse_loss = 0.0
        for ht_i in ht:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                kl_loss += torch.nn.functional.kl_div(torch.log_softmax(hs, dim=1), torch.softmax(ht_i, dim=1))
        for i in range(len(ft_)):
            mse_loss += torch.nn.functional.mse_loss(ft_[i], ft[i])

        return kl_loss + self.beta * mse_loss