# import os
# import sys
# sys.path.append("/data4/tongshuo/Grading/CommonFeatureLearning")
import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller
from ._common import ConvReg, get_feat_shapes
import torchvision.models as models
from torch.hub import tqdm, load_state_dict_from_url as load_url  # noqa: F401
# from cfg import CFG as cfg
# from models.resnet_FitNet import *

class FitNet(Distiller):
    """FitNets: Hints for Thin Deep Nets"""

    def __init__(self, student, teacher, teacher2, cfg, device):
        super(FitNet, self).__init__(student, teacher)
        self.teacher = teacher
        self.teacher2 = teacher2   
        self.ce_loss_weight = cfg.FITNET.LOSS.CE_WEIGHT
        self.feat_loss_weight = cfg.FITNET.LOSS.FEAT_WEIGHT
        self.hint_layer = cfg.FITNET.HINT_LAYER
        feat_s_shapes, feat_t_shapes = get_feat_shapes(
            self.student, self.teacher, cfg.FITNET.INPUT_SIZE, device
        )
        feat_s_shapes, feat_t2_shapes = get_feat_shapes(
            self.student, self.teacher2, cfg.FITNET.INPUT_SIZE, device
        )
        self.conv_reg = ConvReg(
            feat_s_shapes[self.hint_layer], feat_t_shapes[self.hint_layer]
        ).to(device)
        self.conv_reg2 = ConvReg(
            feat_s_shapes[self.hint_layer], feat_t2_shapes[self.hint_layer]
        ).to(device)
    # def get_learnable_parameters(self):
    #     return super().get_learnable_parameters() + list(self.conv_reg.parameters())

    # def get_extra_parameters(self):
    #     num_p = 0
    #     for p in self.conv_reg.parameters():
    #         num_p += p.numel()
    #     return num_p

    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            _, feature_teacher = self.teacher(image)
            _, feature_teacher2 = self.teacher2(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        f_s = self.conv_reg(feature_student["feats"][self.hint_layer])
        loss_feat = self.feat_loss_weight * F.mse_loss(
            f_s, feature_teacher["feats"][self.hint_layer]
        )
        f_s2 = self.conv_reg2(feature_student["feats"][self.hint_layer])
        loss_feat2 = self.feat_loss_weight * F.mse_loss(
            f_s2, feature_teacher2["feats"][self.hint_layer]
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd1": loss_feat,
            "loss_kd2": loss_feat2,            
        }
        return logits_student, losses_dict

# num_classes = 5

# input_data = torch.randn(8, 3, 224, 224)  # 一张224x224的RGB图像
# target = torch.randint(0, num_classes, (8,))

# teacher = resnet50(pretrained=False)  # 加载原始的预训练模型
# teacher.fc = nn.Linear(2048, num_classes)
# teacher.load_state_dict(torch.load('/data4/tongshuo/Grading/CommonFeatureLearning/results/baselines/resnet50_aptos01234_max_acc_ce_notran.pth')['model_state'])

# student = resnet18(pretrained=True)  # 加载原始的预训练模型
# student.fc = nn.Linear(512, num_classes)


# fitnet = FitNet(student, teacher, teacher, cfg)

# logits_student, losses_dict = fitnet.forward_train(input_data, target)

# print('logits_student:', logits_student)
# print('losses_dict:', losses_dict)