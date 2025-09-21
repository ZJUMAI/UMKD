# import os
# import sys
# sys.path.append("/data4/tongshuo/Grading/CommonFeatureLearning")
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
# from models.resnet_FitNet import *
# from cfg import CFG as cfg

def _pdist(e, squared, eps):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


def rkd_loss(f_s, f_t, squared=False, eps=1e-12, distance_weight=25, angle_weight=50):
    stu = f_s.view(f_s.shape[0], -1)
    tea = f_t.view(f_t.shape[0], -1)

    # RKD distance loss
    with torch.no_grad():
        t_d = _pdist(tea, squared, eps)
        mean_td = t_d[t_d > 0].mean()
        t_d = t_d / mean_td

    d = _pdist(stu, squared, eps)
    mean_d = d[d > 0].mean()
    d = d / mean_d

    loss_d = F.smooth_l1_loss(d, t_d)

    # RKD Angle loss
    with torch.no_grad():
        td = tea.unsqueeze(0) - tea.unsqueeze(1)
        norm_td = F.normalize(td, p=2, dim=2)
        t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

    sd = stu.unsqueeze(0) - stu.unsqueeze(1)
    norm_sd = F.normalize(sd, p=2, dim=2)
    s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

    loss_a = F.smooth_l1_loss(s_angle, t_angle)

    loss = distance_weight * loss_d + angle_weight * loss_a
    return loss


class RKD(Distiller):
    """Relational Knowledge Disitllation, CVPR2019"""

    def __init__(self, student, teacher, teacher2, cfg):
        super(RKD, self).__init__(student, teacher)
        self.teacher = teacher
        self.teacher2 = teacher2      
        self.distance_weight = cfg.RKD.DISTANCE_WEIGHT
        self.angle_weight = cfg.RKD.ANGLE_WEIGHT
        self.ce_loss_weight = cfg.RKD.LOSS.CE_WEIGHT
        self.feat_loss_weight = cfg.RKD.LOSS.FEAT_WEIGHT
        self.eps = cfg.RKD.PDIST.EPSILON
        self.squared = cfg.RKD.PDIST.SQUARED

    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            _, feature_teacher = self.teacher(image)
            _, feature_teacher2 = self.teacher2(image)
        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_rkd = self.feat_loss_weight * rkd_loss(
            feature_student["pooled_feat"],
            feature_teacher["pooled_feat"],
            self.squared,
            self.eps,
            self.distance_weight,
            self.angle_weight,
        )
        loss_rkd2 = self.feat_loss_weight * rkd_loss(
            feature_student["pooled_feat"],
            feature_teacher2["pooled_feat"],
            self.squared,
            self.eps,
            self.distance_weight,
            self.angle_weight,
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd1": loss_rkd,
            "loss_kd2": loss_rkd2,
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


# fitnet = RKD(student, teacher, teacher, cfg)

# logits_student, losses_dict = fitnet.forward_train(input_data, target)

# print('logits_student:', logits_student)
# print('losses_dict:', losses_dict)