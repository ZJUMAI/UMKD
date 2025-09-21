from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, cohen_kappa_score
from sklearn.metrics import classification_report
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_params_million = total_params / 1e6
    return total_params, total_params_million

def get_model_ckpts(dataset_name, task_type):
    # 选择模型权重的路径
    if dataset_name == 'eyepacs':
        if task_type == 'balanced_teacher':
            t1_ckpt = '/data4/tongshuo/Grading/CommonFeatureLearning/results/baselines/balanced_teacher/resnet50_eyepacs_max_acc_ce_notran.pth'
            t2_ckpt = '/data4/tongshuo/Grading/CommonFeatureLearning/results/baselines/balanced_teacher/resnet50_eyepacs_1_max_acc_ce_notran.pth'
        else:
            t1_ckpt = '/data4/tongshuo/Grading/CommonFeatureLearning/results/baselines/imbalanced_teacher/resnet50_eyepacs01234_max_acc_ce_notran.pth'
            t2_ckpt = '/data4/tongshuo/Grading/CommonFeatureLearning/results/baselines/imbalanced_teacher/resnet50_eyepacs01234_1_max_acc_ce_notran.pth'
    
    elif dataset_name == 'sicapv2':
        if task_type == 'balanced_teacher':
            t1_ckpt = '/data4/tongshuo/Grading/CommonFeatureLearning/results/baselines/balanced_teacher/resnet50_sicap0123_max_acc_ce_notran.pth'
            t2_ckpt = '/data4/tongshuo/Grading/CommonFeatureLearning/results/baselines/balanced_teacher/resnet50_sicap0123_1_max_acc_ce_notran.pth'
        else:
            t1_ckpt = '/data4/tongshuo/Grading/CommonFeatureLearning/results/baselines/imbalanced_teacher/resnet50_sicap0123_max_acc_ce_notran.pth'
            t2_ckpt = '/data4/tongshuo/Grading/CommonFeatureLearning/results/baselines/imbalanced_teacher/resnet50_sicap0123_1_max_acc_ce_notran.pth'
    
    elif dataset_name == 'aptos':
        if task_type == 'balanced_teacher':
            t1_ckpt = '/data4/tongshuo/Grading/CommonFeatureLearning/results/baselines/balanced_teacher/resnet50_aptos01234_max_acc_ce_notran.pth'
            t2_ckpt = '/data4/tongshuo/Grading/CommonFeatureLearning/results/baselines/balanced_teacher/resnet50_aptos01234_1_max_acc_ce_notran.pth'
        else:
            t1_ckpt = '/data4/tongshuo/Grading/CommonFeatureLearning/results/baselines/imbalanced_teacher/resnet50_aptos01234_max_acc_ce_notran.pth'
            t2_ckpt = '/data4/tongshuo/Grading/CommonFeatureLearning/results/baselines/imbalanced_teacher/resnet50_aptos01234_1_max_acc_ce_notran.pth'
    elif dataset_name == 'aptos01234':
        if task_type == 'balanced_teacher':
            t1_ckpt = '/data4/tongshuo/Grading/CommonFeatureLearning/results/baselines/balanced_teacher/resnet50_aptos01234_max_acc_ce_notran.pth'
            t2_ckpt = '/data4/tongshuo/Grading/CommonFeatureLearning/results/baselines/balanced_teacher/resnet50_aptos01234_1_max_acc_ce_notran.pth'
        else:
            t1_ckpt = '/data4/tongshuo/Grading/CommonFeatureLearning/results/baselines/resnet50_aptos01234_max_acc_ce_notran.pth'
            t2_ckpt = '/data4/tongshuo/Grading/CommonFeatureLearning/results/baselines/resnet50_1_aptos01234_max_acc_ce_notran.pth'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return t1_ckpt, t2_ckpt

def cal_mae_acc_cls(logits, targets, is_sto=True):
    if is_sto:
        r_dim, s_dim, out_dim = logits.shape

        label_arr = torch.arange(0, out_dim).float().cuda()

        probs = F.softmax(logits, -1)

        exp = torch.sum(probs * label_arr, dim=-1)

        exp = torch.mean(exp, dim=0)

        max_a = torch.mean(probs, dim=0)

        max_data = max_a.cpu().data.numpy()

        max_data = np.argmax(max_data, axis=1)
        target_data = targets.cpu().data.numpy()
        exp_data = exp.cpu().data.numpy()
        mae = sum(abs(exp_data - target_data)) * 1.0 / len(target_data)
        acc = sum(np.rint(exp_data) == target_data) * 1.0 / len(target_data)

    else:
        s_dim, out_dim = logits.shape
        probs = F.softmax(logits, -1)
        probs_data = probs.cpu().data.numpy()
        target_data = targets.cpu().data.numpy()
        max_data = np.argmax(probs_data, axis=1)
        label_arr = np.array(range(out_dim))
        exp_data = np.sum(probs_data * label_arr, axis=1)
        mae = sum(abs(exp_data - target_data)) * 1.0 / len(target_data)
        acc = sum(np.rint(exp_data) == target_data) * 1.0 / len(target_data)

    return acc, mae

def accuracy(score, target):
    _, pred = score.max(1)
    # reg_v = torch.round(reg_value)
    # pred = torch.trunc(0.5*(pred+reg_v))
    # equal = (pred == target).float()
    acc = accuracy_score(target.cpu().data.numpy(), pred.cpu().data.numpy())  # the same as equal.mean().data[0]

    # print('equal',equal,equal.mean())
    return acc

def specific_eval(score, target):
    _, pred = score.max(1)
    prediction = pred.cpu().data.numpy()
    true_lable = target.cpu().data.numpy()
    # print(pred, prediction, prediction.shape, true_lable)
    # precision_score_average_None = precision_score(true_lable, prediction, average=None)
    # precision_score_average_micro = precision_score(true_lable, prediction, average='micro')
    # precision_score_average_macro = precision_score(true_lable, prediction, average='macro')
    # precision_score_average_weighted = precision_score(true_lable, prediction, average='weighted')
    measure_result = classification_report(true_lable, prediction)
    # print('measure_result = \n', measure_result)
    measure_dict = classification_report(true_lable, prediction, output_dict=True)

    return measure_dict, measure_result

def Arg_MAE(score, target):
    _, pred = score.max(1)
    mae = sum((pred-target)**2)
    return mae

def MAE(score, target):
    # _, pred = score.max(1)
    # l1loss = nn.L1Loss()
    # target = target.float()
    # mae = l1loss(pred, target)

    s_dim, out_dim = score.shape
    probs = F.softmax(score, -1)
    probs_data = probs.cpu().data.numpy()
    target_data = target.cpu().data.numpy()
    max_data = np.argmax(probs_data, axis=1)
    label_arr = np.array(range(out_dim))
    exp_data = np.sum(probs_data * label_arr, axis=1)
    mae = sum(abs(exp_data - target_data)) * 1.0 / len(target_data)


    return mae

def seperate_acc(score, target):
    o4, o2, o1 = score
    _, o4 = o4.squeeze().max(1)
    _, o2 = o2.squeeze().max(1)
    _, o1 = o1.squeeze().max(1)

    target = target.float()
    l4 = (target // 4).unsqueeze(dim=1)
    l2 = ((target % 4) // 2).unsqueeze(dim=1)
    l1 = (target % 2).unsqueeze(dim=1)

    l1loss = nn.L1Loss()

    acc_4 = (o4 == l4).float()
    acc_2 = (o2 == l2).float()
    acc_1 = (o1 == l1).float()

    acc = (o4 * 4 + o2 * 2 + o1 == l4 * 4 + l2 * 2 + l1).float()

    mae = l1loss(target, target)# 没用的

    return acc_4.mean().item(), acc_2.mean().item(), acc_1.mean().item(), acc.mean().item(),mae

def accuracy_new(score, target):
    _, p4 = score[0].squeeze().max(1)
    _, p2 = score[1].squeeze().max(1)
    _, p1 = score[2].squeeze().max(1)
    pred = p1 + p2 * 2 + p4 * 4
    # target = target[0] * 4 + target[1] * 2 + target[2] * 1

    l1loss = nn.L1Loss()
    target = target.float()
    mae = l1loss(pred, target)

    equal = (pred == target).float()
    # acc = accuracy_score(target.cpu().data.numpy(), pred.cpu().data.numpy())  # the same as equal.mean().data[0]
    # print('equal',equal,equal.mean())
    return equal.mean().item(), mae


import torch
def predictive_entropy(probs):  # probs: [B, C]
    probs = torch.softmax(probs, dim=1)
    return -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean()


from sklearn.metrics import mutual_info_score
def mutual_information(probs, labels):
    pred_labels = torch.argmax(probs, dim=1)
    # 使用sklearn的MI计算（基于直方图统计）
    return mutual_info_score(labels.cpu().numpy(), pred_labels.cpu().numpy())