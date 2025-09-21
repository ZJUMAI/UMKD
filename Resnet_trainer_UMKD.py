#######################
#两个resnet50蒸馏给学生18
#######################
import random
import os, sys
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
from torch.utils import data

from loss import SoftCELoss, CFLoss, CFLoss_SA, euclidean_loss
from utils.stream_metrics import StreamClsMetrics, AverageMeter
from utils.metric import *
from models.cfl import CFL_ConvBlock
from datasets import StanfordDogs, CUB200, DRDataset, APTOSDataset, SICAPv2Dataset
from utils import mkdir_if_missing, Logger
from dataloader import get_concat_dataloader
from torchvision import transforms
from models.resnet_SDD_REDL_multi import resnet18_sdd_redl, resnet34_sdd_redl, resnet50_sdd_redl
from models.resnet_SDD_LP import resnet18_sdd_lp, resnet34_sdd_lp, resnet50_sdd_lp
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
from loss import dkd_no_labels_loss, dkd_loss, multi_dkd, uc_multi_dkd
import time
import re
from utils.metric import predictive_entropy, mutual_information

_model_dict = {
    'resnet18': resnet18_sdd_lp,
    'resnet34': resnet34_sdd_redl,
    'resnet50': resnet50_sdd_redl,
}

alpha_dkd = 1
beta_dkd = 8
temperature_dkd = 4

alpha = 10
beta = 1

class LowPassModule(nn.Module):
    def __init__(self, in_channel, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(size) for size in sizes])
        self.relu = nn.ReLU()
        ch =  in_channel // 4
        self.channel_splits = [ch, ch, ch, ch]
    def _make_stage(self, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        return nn.Sequential(prior)
    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        feats = torch.split(feats, self.channel_splits, dim=1) 
        priors = [F.upsample(input=self.stages[i](feats[i]), size=(h, w), mode='bilinear') for i in range(4)] 
        bottle = torch.cat(priors, 1)
        return self.relu(bottle)

def count_parameters(model):
    # 统计模型所有可训练参数的总数
    total_params = sum(p.numel() for p in model.parameters())
    # 将总参数量转换为百万（M）
    total_params_million = total_params / 1e6
    return total_params, total_params_million

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='/data4/tongshuo/Grading/CommonFeatureLearning/data')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model", type=str, default='resnet18')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu_id", type=str, default='5')
    parser.add_argument("--random_seed", type=int, default=1337)
    parser.add_argument("--download", action='store_true', default=False)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--cfl_lr", type=float, default=None)
    # parser.add_argument("--t1_ckpt", type=str, default='/data4/tongshuo/Grading/CommonFeatureLearning/results/baselines/resnet50_sicap0123_max_acc_ce_notran.pth')
    # parser.add_argument("--t2_ckpt", type=str, default='/data4/tongshuo/Grading/CommonFeatureLearning/results/baselines/resnet50_1_sicap0123_max_acc_ce_notran.pth')
    # parser.add_argument("--t1_ckpt", type=str, default='/data4/tongshuo/Grading/CommonFeatureLearning/results/baselines/resnet50_aptos01234_max_acc_ce_notran.pth')
    # parser.add_argument("--t2_ckpt", type=str, default='/data4/tongshuo/Grading/CommonFeatureLearning/results/baselines/resnet50_1_aptos01234_max_acc_ce_notran.pth')
    # parser.add_argument("--t1_ckpt", type=str, default='/data4/tongshuo/Grading/CommonFeatureLearning/results/baselines/balanced_teacher/resnet50_aptos01234_max_acc_ce_notran.pth')
    # parser.add_argument("--t2_ckpt", type=str, default='/data4/tongshuo/Grading/CommonFeatureLearning/results/baselines/balanced_teacher/resnet50_aptos01234_1_max_acc_ce_notran.pth')
    parser.add_argument("--dataset", type=str, default='aptos',
                        choices=['eyepacs', 'aptos', 'sicapv2', 'aptos01234'])
    parser.add_argument("--task_type", type=str, default='balanced_teacher',
                        choices=['balanced_kd', 'balanced_teacher'])
    parser.add_argument("--layer", type=str, default='layer1',
                        choices=['layer1', 'layer2', 'layer3', 'layer4'])
    return parser

def amal(cur_epoch, criterion, criterion_ce, criterion_cf, criterion_cf_LP, t1_low_pass, t2_low_pass, model, cfl_blk, cfl_blk_SA, teachers, optim, train_loader, device, scheduler=None, print_interval=2):
    """Train and return epoch loss"""
    t1, t2 = teachers
    #if scheduler is not None:
    #    scheduler.step()

    print("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))
    avgmeter = AverageMeter()
    #is_densenet = isinstance(model, DenseNet)
    is_densenet = False
    for cur_step, (images, labels) in enumerate(train_loader):

        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)

        # get soft-target logits
        optim.zero_grad()
        with torch.no_grad():
            t1_out, patch_score_t1, uc1 = t1(images, uncertainty_type='max_modified_prob', lamb1=1.0, lamb2=0.1, score_norm=False)
            t2_out, patch_score_t2, uc2 = t2(images, uncertainty_type='max_modified_prob', lamb1=1.0, lamb2=0.1, score_norm=False)
            # t_outs = torch.cat((t1_out, t2_out), dim=1)
            uc1 = np.array(uc1)
            uc2 = np.array(uc2)
            uc_mean = (uc1 + uc2) / 2
            stacked_output = torch.stack((patch_score_t1, patch_score_t2), dim=0)
            patch_t_outs = torch.mean(stacked_output, dim=0)
###############################################################################            
            ft1_SA = t1.layer1.output
            ft2_SA = t2.layer1.output
###############################################################################
            ft1 = t1.layer4.output
            ft2 = t2.layer4.output

        # get student output
        s_outs, patch_score_s, fs_LP = model(images)
        if is_densenet:
            fs = model.features.output
        else:
            fs = model.layer4.output
###############################################################################  
        ft1_LP = t1_low_pass(ft1_SA)
        ft2_LP = t2_low_pass(ft2_SA)
        ft_LP = [ft1_LP, ft2_LP]
        (hs_LP, ht_LP), (ft_LP_, ft_LP) = cfl_blk_SA(fs_LP, ft_LP)
###############################################################################
        ft = [ft1, ft2]
        (hs, ht), (ft_, ft) = cfl_blk(fs, ft)

        loss_1 = criterion(s_outs, labels)  #输出与真实标签之间计算损失
        # loss_ce = criterion_ce(s_outs, t_outs) #软目标损失
########将损失替换为DKD_loss#######
        # loss_tckd, loss_nckds = dkd_loss(s_outs, t_outs, labels, alpha, beta, temperature)
        # loss_ce = loss_tckd + loss_nckds
########将损失替换为SDD_DKD_loss#######
        # loss_ce = multi_dkd(patch_score_s, patch_t_outs, labels, alpha, beta, temperature)
########将损失替换为UDD_DKD_loss#######
        # loss_ce_t1 = uc_multi_dkd(patch_score_s, patch_score_t1, labels, alpha, beta, temperature, uc1)
        # loss_ce_t2 = uc_multi_dkd(patch_score_s, patch_score_t2, labels, alpha, beta, temperature, uc2)
        # loss_ce = loss_ce_t1 + loss_ce_t2
        loss_ce = uc_multi_dkd(patch_score_s, patch_t_outs, labels, alpha_dkd, beta_dkd, temperature_dkd, uc_mean)
        loss_cf = criterion_cf(hs, ht, ft_, ft) #MMD和重构损失
        loss_cf_LP = criterion_cf_LP(hs_LP, ht_LP, ft_LP_, ft_LP) #浅层特征的MMD损失,没有重构损失
        loss = loss_1 + alpha*(loss_cf + loss_cf_LP) + beta*loss_ce
        loss.backward()
        optim.step()

        avgmeter.update('loss', loss.item())
        avgmeter.update('interval loss', loss.item())
        avgmeter.update('ce loss', loss_ce.item())
        avgmeter.update('cf loss', loss_cf.item())
        
        if (cur_step+1) % print_interval == 0:
            interval_loss = avgmeter.get_results('interval loss')
            ce_loss = avgmeter.get_results('ce loss')
            cf_loss = avgmeter.get_results('cf loss')
            acc = accuracy(s_outs, labels)
            mae = MAE(s_outs, labels)
            print("Epoch %d, Batch %d/%d, Loss=%f (cls=%f, sdd_loss=%f, cf=%f, cf_LP=%f), Acc=%f, MAE=%f"  %
                  (cur_epoch, cur_step+1, len(train_loader), interval_loss, loss_1, loss_ce, loss_cf, loss_cf_LP, acc, mae))
            avgmeter.reset('interval loss')
            avgmeter.reset('ce loss')
            avgmeter.reset('cf loss')
    if scheduler is not None:
        scheduler.step()
    # return avgmeter.get_results('loss')


def validate(model, loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()
    total_logit = []
    total_target = []
    failed_cases = []
    failed_images = []
    inference_times = []  # 用于记录每个批次的推理时长

    with torch.no_grad():
        for idx, (images, labels) in tqdm(enumerate(loader)):
            # 记录推理开始时间
            start_time = time.time()

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs, patch_score_s, fs_LP = model(images)

            # 记录推理结束时间
            end_time = time.time()
            inference_time = end_time - start_time  # 单个批次的推理时长
            inference_times.append(inference_time)

            preds = outputs.detach()  # .max(dim=1)[1].cpu().numpy()
            targets = labels  # .cpu().numpy()
            for i in range(len(labels)):  # 遍历 batch_size 维度
                predicted = outputs.max(dim=1)[1].cpu().numpy()
                label = labels.cpu().numpy()
                if predicted[i] != label[i]:  # 判断是否预测错误
                    global_idx = idx * len(labels) + i
                    failed_cases.append((global_idx, label[i].item(), predicted[i].item()))
                    img_path = loader.dataset.data_list[global_idx][0]  # 读取图像路径
                    failed_images.append(img_path)
            if idx == 0:
                total_logit = preds
                total_target = targets
            else:
                total_logit = torch.cat([total_logit, preds], dim=0)
                total_target = torch.cat([total_target, targets], 0)
            metrics.update(preds, targets)
        score = metrics.get_results()
        mae = MAE(total_logit, total_target)
        entropy = predictive_entropy(total_logit)
        mi = mutual_information(total_logit, total_target)
        score['MAE'] = mae
        dic_data, measure_result = specific_eval(total_logit, total_target)
        val_f1 = dic_data["weighted avg"]["f1-score"]
        val_recall = dic_data["macro avg"]["recall"]
        val_precision = dic_data["macro avg"]["precision"]
        
    return score, val_f1, val_recall, val_precision, measure_result, failed_cases, failed_images, entropy, mi


def main():
    opts = get_parser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t1_ckpt, t2_ckpt = get_model_ckpts(opts.dataset, opts.task_type)
    print(t1_ckpt, t2_ckpt)
    # Set up random seed
    mkdir_if_missing('checkpoints')
    mkdir_if_missing('logs')
    sys.stdout = Logger(os.path.join('logs', 'amal_%s.txt'%(opts.model)))
    print(opts)
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    if opts.dataset == 'aptos':
        dataset = APTOSDataset
        num_classes = 5
    elif opts.dataset == 'aptos01234':
        dataset = APTOSDataset
        num_classes = 5
    elif opts.dataset == 'sicapv2':
        dataset = SICAPv2Dataset
        num_classes = 4
    elif opts.dataset == 'eyepacs':
        dataset = DRDataset
        num_classes = 5

    cur_epoch = 0
    max_acc = 0.0
    min_mae = 1000
    max_f1 = 0.0
    max_mean_acc = 0.0
    max_mi = 0.0
    min_en = 1000

    mkdir_if_missing('checkpoints')
    macc_ckpt = "/data4/tongshuo/Grading/CommonFeatureLearning/ckp_rebuttal/%s_%s_macc_UMKD.pth"%(opts.dataset, opts.task_type)
    # max_acc_ckpt = '/data4/tongshuo/Grading/CommonFeatureLearning/results/student/%s/%s/%s_%s_%s_max_acc_3tckd_1_unckd.pth'%(opts.dataset, opts.task_type, opts.model, opts.dataset, opts.layer)
    # min_mae_ckpt = '/data4/tongshuo/Grading/CommonFeatureLearning/results/student/%s/%s/%s_%s_%s_min_mae_3tckd_1_unckd.pth'%(opts.dataset, opts.task_type, opts.model, opts.dataset, opts.layer)
    #  Set up dataloader
    #train_loader, val_loader = get_concat_dataloader(data_root=opts.data_root, batch_size=opts.batch_size, download=opts.download)
    tran = transforms.Compose([
            transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            # transforms.RandomApply([
            #     transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            # ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    if opts.task_type == 'balanced_kd':
        train_data = dataset(None,None,'train',tran,0, True)
        val_data = dataset(None,None,'valid',tran,0, True)
    else:
        train_data = dataset(None,None,'train',tran,0, False)
        val_data = dataset(None,None,'valid',tran,0, False)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=opts.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=opts.batch_size, shuffle=False, drop_last=True)

    # pretrained teachers
    t1_model_name = 'resnet50'
    t1 = _model_dict[t1_model_name](num_classes=num_classes, M = '[1,2,4]').to(device)
    t2_model_name = 'resnet50'
    t2 = _model_dict[t2_model_name](num_classes=num_classes, M = '[1,2,4]').to(device)
    print("Loading pretrained teachers ...\nT1: %s, T2: %s"%(t1_model_name, t2_model_name))
    t1.load_state_dict(torch.load(t1_ckpt)['model_state'])
    t2.load_state_dict(torch.load(t2_ckpt)['model_state'])
    t1.eval()
    t2.eval()
    print("Target student: %s"%opts.model)
    stu = _model_dict[opts.model](pretrained=True, num_classes=num_classes, M = '[1,2,4]').to(device)
    metrics = StreamClsMetrics(num_classes)
##############################################
    # 打印模型的总参数量
    total_params, total_params_million = count_parameters(stu)
    print(f"Total number of parameters: {total_params} ({total_params_million:.2f}M)")
##############################################
    # Setup Common Feature Blocks
    t1_feature_dim = t1.fc.in_features #512
    t2_feature_dim = t2.fc.in_features #512
    is_densenet = True if 'densenet' in opts.model else False
    if is_densenet:
        stu_feature_dim = stu.classifier.in_features
    else:
        stu_feature_dim = stu.fc.in_features #512
    cfl_blk = CFL_ConvBlock(stu_feature_dim, [t1_feature_dim, t2_feature_dim], 128).to(device)
    def forward_hook(module, input, output):
        module.output = output # keep feature maps
    t1.layer4.register_forward_hook(forward_hook)
    t2.layer4.register_forward_hook(forward_hook)
    if is_densenet:
        stu.features.register_forward_hook(forward_hook)
    else:
        stu.layer4.register_forward_hook(forward_hook)
##########################################################################################################
###############添加浅层的MMD损失###################
    t1_feature_dim_SA = t1.layer2[0].conv1.in_channels #256
    t2_feature_dim_SA = t2.layer2[0].conv1.in_channels #256
    t1_low_pass = LowPassModule(in_channel = t1_feature_dim_SA)
    t2_low_pass = LowPassModule(in_channel = t2_feature_dim_SA)
    is_densenet = True if 'densenet' in opts.model else False
    if is_densenet:
        stu_feature_dim_SA = stu.classifier.in_features
    else:
        stu_feature_dim_SA = stu.layer2[0].conv1.in_channels #64
        # stu_feature_dim_SA = stu.fc.in_features #512

    cfl_blk_SA = CFL_ConvBlock(stu_feature_dim_SA, [t1_feature_dim_SA, t2_feature_dim_SA], 128).to(device)
    def forward_hook(module, input, output):
        module.output = output # keep feature maps
    t1.layer1.register_forward_hook(forward_hook)
    t2.layer1.register_forward_hook(forward_hook)
    if is_densenet:
        stu.features.register_forward_hook(forward_hook)
    else:
        stu.layer1.register_forward_hook(forward_hook)
##########################################################################################################
    params_1x = []
    params_10x = []
    for name, param in stu.named_parameters():
        if 'fc' in name:
            params_10x.append(param)
        else:
            params_1x.append(param)

    cfl_lr = opts.lr*10 if opts.cfl_lr is None else opts.cfl_lr
    optimizer = torch.optim.Adam([{'params': params_1x,             'lr': opts.lr},
                                  {'params': params_10x,            'lr': opts.lr*10},
                                  {'params': cfl_blk.parameters(),  'lr': cfl_lr}, 
                                  {'params': cfl_blk_SA.parameters(),  'lr': cfl_lr}, ],
                                 lr=opts.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=15, gamma=0.1)

##### Loss #####
    criterion = nn.CrossEntropyLoss(reduction='mean')
    criterion_ce = SoftCELoss(T=1.0)
    criterion_cf = CFLoss(normalized=True)
    criterion_cf_LP = CFLoss(normalized=True)
    # criterion_cf_LP = CFLoss_SA(normalized=True)

    def save_ckpt(path):
        """ save current model
        """
        state = {
            "epoch": cur_epoch,
            "model_state": stu.state_dict(),
            "cfl_state": cfl_blk.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "max_acc": max_acc,
            "min_mae": min_mae
        }
        torch.save(state, path)
        print("Model saved as %s" % path)

    print("Training ...")
    # ===== Train Loop =====#
    while cur_epoch < opts.epochs:
        stu.train()
        amal(cur_epoch=cur_epoch,
                            criterion=criterion,
                            criterion_ce=criterion_ce,
                            criterion_cf=criterion_cf,
                            criterion_cf_LP=criterion_cf_LP,
                            t1_low_pass = t1_low_pass,
                            t2_low_pass = t2_low_pass,
                            model=stu,
                            cfl_blk=cfl_blk,
                            cfl_blk_SA=cfl_blk_SA,
                            teachers=[t1, t2],
                            optim=optimizer,
                            train_loader=train_loader,
                            device=device,
                            scheduler=scheduler)
        # print("End of Epoch %d/%d, Average Loss=%f" %
        #      (cur_epoch, opts.epochs, epoch_loss))

        # =====  Latest Checkpoints  =====
        # save_ckpt(latest_ckpt)
        # =====  Validation  =====
        print("validate on val set...")
        stu.eval()
        val_score, val_f1, val_recall, val_precision, measure_result, failed_cases, failed_images, entropy, mi = validate(model=stu,
                             loader=val_loader,
                             device=device,
                             metrics=metrics)
        print("Confusion Matrix:\n",val_score['Confusion Matrix'])
        print(metrics.to_str(val_score))
        # dataset_name = re.sub(r'\d', '', opts.dataset)
        # score_path = '/data4/tongshuo/Grading/CommonFeatureLearning/results/student/%s/%s/%s_%s_%s_ce_UKA_2tckd_1_unckd.txt'%(opts.dataset, opts.task_type, opts.model, opts.dataset, opts.layer)
        # score_best_path = '/data4/tongshuo/Grading/CommonFeatureLearning/results/student/%s/%s/%s_%s_%s_best_ce_UKA_2tckd_1_unckd.txt'%(opts.dataset, opts.task_type, opts.model, opts.dataset, opts.layer)
        score_path = '/data4/tongshuo/Grading/CommonFeatureLearning/result_rebuttal/%s_%s_UMKD.txt'%(opts.dataset, opts.task_type)
        score_best_path = '/data4/tongshuo/Grading/CommonFeatureLearning/result_rebuttal/%s_%s_best_UMKD.txt'%(opts.dataset, opts.task_type)
        
        with open(score_path, mode='a') as f:
            f.write("^^^^^^^^Epoch %d:\n"%cur_epoch)
            f.write("Confusion Matrix:\n")
            f.write(str(val_score['Confusion Matrix']) + "\n") 
            f.write(str(measure_result) + "\n")  
            f.write(metrics.to_str(val_score)+ "\n")
            # 保存 failed_images
            f.write("Failed Images:")
            f.write(str(failed_images) + "\n") 
            f.write("Failed Cases:")
            f.write(str(failed_cases) + "\n")
        sys.stdout.flush()

        # =====  Save Best Model  =====
        if val_score['Overall Acc'] > max_acc:  # save best model
            max_acc = val_score['Overall Acc']
            # save_ckpt(max_acc_ckpt)

            with open(score_best_path, mode='a') as f:
                f.write("Epoch %d: acc_mae_f1_mean_en_mi = [%f, %f, %f, %f, %f, %f]\n" % (cur_epoch, max_acc, min_mae, max_f1, max_mean_acc, min_en, max_mi))


        if val_score['MAE'] < min_mae:  # save best model
            min_mae = val_score['MAE']
            # save_ckpt(min_mae_ckpt)

            with open(score_best_path, mode='a') as f:
                f.write("Epoch %d: mae_acc_f1_mean_en_mi = [%f, %f, %f, %f, %f, %f]\n" % (cur_epoch, min_mae, max_acc, max_f1, max_mean_acc, min_en, max_mi))

        if val_f1 > max_f1:  # save best model
            max_f1 = val_f1
            # save_ckpt(min_mae_ckpt)

            with open(score_best_path, mode='a') as f:
                f.write("Epoch %d: f1_acc_mae_mean_en_mi = [%f, %f, %f, %f, %f, %f]\n" % (cur_epoch, max_f1, max_acc, min_mae, max_mean_acc, min_en, max_mi))


        if val_score['Mean Acc'] > max_mean_acc:  # save best model
            max_mean_acc = val_score['Mean Acc']
            save_ckpt(macc_ckpt)

            with open(score_best_path, mode='a') as f:
                f.write("Epoch %d: mean_acc_mae_f1_en_mi = [%f, %f, %f, %f, %f, %f]\n" % (cur_epoch, max_mean_acc, max_acc, min_mae, max_f1, min_en, max_mi))


        if entropy < min_en:  # save best model
            min_en = entropy
            # save_ckpt(min_mae_ckpt)
            with open(score_best_path, mode='a') as f:
                f.write("Epoch %d: en_mean_acc_mae_f1_mi = [%f, %f, %f, %f, %f, %f]\n" % (cur_epoch, min_en, max_mean_acc, max_acc, min_mae, max_f1, max_mi))


        if mi > max_mi:  # save best model
            max_mi = mi
            # save_ckpt(min_mae_ckpt)
            with open(score_best_path, mode='a') as f:
                f.write("Epoch %d: mi_mean_acc_mae_f1_en = [%f, %f, %f, %f, %f, %f]\n" % (cur_epoch, max_mi, max_mean_acc, max_acc, min_mae, max_f1, min_en))
        # if val_score['Overall Acc'] > max_acc:  # save best model
        #     max_acc = val_score['Overall Acc']
        #     save_ckpt(max_acc_ckpt)

        #     with open(score_best_path, mode='a') as f:
        #         f.write("Epoch %d:\n"%cur_epoch)
        #         f.write("Confusion Matrix:\n")
        #         f.write(str(val_score['Confusion Matrix']) + "\n") 
        #         f.write(str(measure_result) + "\n")
        #         f.write(metrics.to_str(val_score))

        # if val_score['MAE'] < min_mae:  # save best model
        #     min_mae = val_score['MAE']
        #     # save_ckpt(min_mae_ckpt)

        #     with open(score_best_path, mode='a') as f:
        #         f.write("Epoch %d:\n"%cur_epoch)
        #         f.write("Confusion Matrix:\n")
        #         f.write(str(val_score['Confusion Matrix']) + "\n")   
        #         f.write(str(measure_result) + "\n")
        #         f.write(metrics.to_str(val_score))
        # if val_f1 > max_f1:  # save best model
        #     max_f1 = val_f1
        #     # save_ckpt(min_mae_ckpt)

        #     with open(score_best_path, mode='a') as f:
        #         f.write("Epoch %d:\n"%cur_epoch)
        #         f.write("Confusion Matrix:\n")
        #         f.write(str(val_score['Confusion Matrix']) + "\n")   
        #         f.write(str(measure_result) + "\n")
        #         f.write(metrics.to_str(val_score))

        # if val_score['Mean Acc'] > max_mean_acc:  # save best model
        #     max_mean_acc = val_score['Mean Acc']
        #     # save_ckpt(min_mae_ckpt)

        #     with open(score_best_path, mode='a') as f:
        #         f.write("Epoch %d:\n"%cur_epoch)
        #         f.write("Confusion Matrix:\n")
        #         f.write(str(val_score['Confusion Matrix']) + "\n")   
        #         f.write(str(measure_result) + "\n")
        #         f.write(metrics.to_str(val_score))
        
        cur_epoch += 1

if __name__ == '__main__':
    main()
