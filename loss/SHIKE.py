import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
# 忽略所有警告
warnings.filterwarnings('ignore')
# from ._base import Distiller

""" 
logits_student : 学生网络的逻辑输出
logits_teacher : 教师网络的逻辑输出
target ：标签值
alpha、beta、temperature : 超参数
"""
def SHIKE_dkd_no_labels_loss(logits_student, logits_teacher, alpha, beta, temperature, device):
    logits_rank = logits_teacher[0].unsqueeze(1)
    for i in range(len(logits_teacher) - 1):
        logits_rank = torch.cat(
            (logits_rank, logits_teacher[i+1].unsqueeze(1)), dim=1)# print('$$$',logits_rank, logits_rank.shape)###形式转换为每个样本，若干个专家的logits输出
    avg_logits_teacher = torch.sum(logits_rank, dim=1) / len(logits_teacher)###计算每个类别在4个专家输出中的均值
    # print('avg_logits_teacher:',avg_logits_teacher)
    _, target = torch.max(avg_logits_teacher, dim=1)
    target_onehot = torch.zeros((target.size(0), logits_student.shape[-1]), device=device)
    target_onehot[torch.arange(target.size(0)), target] = 1
    # print('target:',target, target.shape[0])
    # print('target_onehot:',target_onehot)
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    # print('$$$$$$$$gt_mask:',gt_mask)
    # print('$$$$$$$$ther_mask:',other_mask)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(avg_logits_teacher / temperature, dim=1)
    # print('#####pred_student:',pred_student)
    # print('#####pred_teacher:',pred_teacher)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    # print('pred_student:',pred_student)
    # print('pred_teacher:',pred_teacher)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    # print('tckd_loss:',tckd_loss)

#########这里对于NCKD的计算可以参考削弱最困难负类的做法###############
    # log_pred_student_part2 = F.log_softmax(
    #     logits_student / temperature - 1000.0 * gt_mask, dim=1
    # )
    # pred_teacher_part2 = F.softmax(
    #     avg_logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    # )
    # nckd_loss = (
    #     F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
    #     * (temperature**2)
    #     / target.shape[0]
    # )
    nckd_loss = SHIKE(logits_student, logits_teacher, target_onehot, temperature, balance=False, label_dis=None)
    # print('nckd_loss:',nckd_loss)
#########这里对于NCKD的计算可以参考削弱最困难负类的做法###############
    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

def SHIKE(logits_student, logits_teacher, labels, temperature, balance=False, label_dis=None):
    logits_rank = logits_teacher[0].unsqueeze(1)
    for i in range(len(logits_teacher) - 1):
        logits_rank = torch.cat(
            (logits_rank, logits_teacher[i+1].unsqueeze(1)), dim=1)
    max_tea, max_idx = torch.max(logits_rank, dim=1) #找到每个类别在三个专家中的最大概率
    non_target_labels = torch.ones_like(labels) - labels
    avg_logits = torch.sum(logits_rank, dim=1) / len(logits_teacher)###计算每个类别在三个专家输出中的均值
    non_target_logits = (-30 * labels) + avg_logits * non_target_labels
    _hardest_nt, hn_idx = torch.max(non_target_logits, dim=1)#把找到最难非负类别的索引
    hardest_idx = torch.zeros_like(labels)
    hardest_idx.scatter_(1, hn_idx.data.view(-1, 1), 1) #将最难非负类索引转换为one-hot
    hardest_logit = non_target_logits * hardest_idx #只保留最难非负类的logist,其余变为0
    rest_nt_logits = max_tea * (1 - hardest_idx) * (1 - labels) ###得到剩余类别中每个类别的最大概率值
    reformed_nt = rest_nt_logits + hardest_logit #将最困难样本的均值logit与剩余类别在所有专家中的最大logist拼接为新的NCKD
    
    preds = F.softmax(logits_student)  #给学生模型的输出做了个softmax
    student_preds = preds * labels
    student_preds = torch.sum(student_preds, dim=-1, keepdim=True)
    student_min = -30 * labels
    student_excluded_preds = F.softmax(
            (logits_student / temperature) * (1 - labels) + student_min)
    # print('student_excluded_preds:',student_excluded_preds)
    # print('reformed_nt:',reformed_nt)
    loss = F.kl_div(torch.log(student_excluded_preds), F.softmax(reformed_nt / temperature))
    return loss


# class DKD(Distiller):
#     """Decoupled Knowledge Distillation(CVPR 2022)"""

#     def __init__(self, student, teacher, cfg):
#         super(DKD, self).__init__(student, teacher)
#         self.ce_loss_weight = cfg.DKD.CE_WEIGHT
#         self.alpha = cfg.DKD.ALPHA
#         self.beta = cfg.DKD.BETA
#         self.temperature = cfg.DKD.T
#         self.warmup = cfg.DKD.WARMUP

#     def forward_train(self, image, target, **kwargs):
#         logits_student, _ = self.student(image)
#         with torch.no_grad():
#             logits_teacher, _ = self.teacher(image)

#         # losses
#         loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
#         loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) * dkd_loss(
#             logits_student,
#             logits_teacher,
#             target,
#             self.alpha,
#             self.beta,
#             self.temperature,
#         )
#         losses_dict = {
#             "loss_ce": loss_ce,
#             "loss_kd": loss_dkd,
#         }
#         return logits_student, losses_dict



# 假设我们有5个模型的输出，每个模型的输出形状为 [batch_size, num_classes]
# device = 'cuda'
# batch_size = 2
# num_classes = 5
# logits_teacher = [torch.randn(batch_size, num_classes).to(device) for _ in range(4)]
# logits_student = torch.randn(batch_size, num_classes).to(device)
# print('logits_teacher:',logits_teacher)

# # 是否使用平衡权重
# balance = False
# alpha = 1
# beta = 8
# temperature = 4

# # 调用函数
# loss = SHIKE_dkd_no_labels_loss(logits_student, logits_teacher, alpha, beta, temperature, device)

# # 打印结果
# print("Loss:", loss)