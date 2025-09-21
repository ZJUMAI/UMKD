# import os
# import sys
# sys.path.append("/data4/tongshuo/Grading/CommonFeatureLearning")
import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.cfl import CFL_ConvBlock

class ContrastiveCFLoss(nn.Module):
    """对比学习损失函数，支持多个教师和学生特征之间的对比学习"""
    def __init__(self, temperature=0.1, normalized=True):
        super(ContrastiveCFLoss, self).__init__()
        self.temperature = temperature
        self.normalized = normalized

    def forward(self, hs, ht_list):
        """
        计算学生特征与多个教师特征之间的对比学习损失

        Args:
            hs (torch.Tensor): 学生特征 [N, D]
            ht_list (List[torch.Tensor]): 多个教师的特征列表 [M][N, D]

        Returns:
            torch.Tensor: 对比学习损失
        """
        # 特征归一化
        if self.normalized:
            hs = F.normalize(hs, p=2, dim=1)
            ht_list = [F.normalize(ht, p=2, dim=1) for ht in ht_list]

        # 计算对比学习损失
        loss = 0.0
        num_teachers = len(ht_list)
        for ht in ht_list:
            # 计算学生与当前教师的相似度
            sim_student_teacher = torch.matmul(hs, ht.t()) / self.temperature  # [N, N]
            logits = F.log_softmax(sim_student_teacher, dim=1)

            # 目标是对角线为1的矩阵（学生与教师特征对齐）
            targets = torch.arange(hs.size(0)).to(hs.device)  # [N]
            loss += F.cross_entropy(logits, targets)

        # 平均所有教师的损失
        return loss / num_teachers


class Sup_ContrastiveCFLoss(nn.Module):
    """对比学习损失函数，支持多个教师和学生特征之间的对比学习"""
    def __init__(self, temperature=0.1, normalized=True):
        super(Sup_ContrastiveCFLoss, self).__init__()
        self.temperature = temperature
        self.normalized = normalized

    def forward(self, hs, ht_list, labels, num_views=2):
        device = hs.device
        batch_size = len(labels)
        """
        计算学生特征与多个教师特征之间的对比学习损失
        """
        # 维度进行转换[N, C, H, W] 变为 [N, D]
        hs = torch.flatten(hs, start_dim=1)
        ht_list = [torch.flatten(feat, start_dim=1) for feat in ht_list]

        # 特征归一化
        if self.normalized:
            hs = F.normalize(hs, p=2, dim=1)
            ht_list = [F.normalize(ht, p=2, dim=1) for ht in ht_list]

        labels=torch.cat([labels.repeat(1), labels.repeat(1)])
        inter_loss = 0.0
        for tea_proj in ht_list:
            features=torch.cat([hs, tea_proj], dim=0)
            # 构建标签矩阵
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)
            # 相似度矩阵
            similarity = torch.matmul(features, features.T) / self.temperature
            # 排除对角线
            logits_mask = torch.ones_like(mask) - torch.eye(len(mask)).to(device)
            mask = mask * logits_mask
            # 计算log_prob
            exp_logits = torch.exp(similarity) * logits_mask
            log_prob = similarity - torch.log(exp_logits.sum(dim=1, keepdim=True))
            # 计算每个正样本对的平均log概率
            per_instance_loss = - (mask * log_prob).sum(1) / mask.sum(1)
            # 按视图数调整损失计算
            if num_views > 1:
                per_instance_loss = per_instance_loss.view(batch_size, num_views).mean(dim=1)
            inter_loss += per_instance_loss.mean()
        # 平均所教师的损失
        return inter_loss / len(ht_list)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

class SupervisedContrastiveCFL(nn.Module):
    def __init__(self, temperature=0.1, projection_dim=128):
        super().__init__()
        self.temperature = temperature
        self.projection = nn.Sequential(
            nn.Linear(2048, 512),  # 假设原始特征维度2048
            nn.ReLU(),
            nn.Linear(512, projection_dim))
        
    def forward(self, stu_features, tea_features_list, labels, num_views=2):
        """
        stu_features: 学生特征 [B*V, D]
        tea_features_list: 教师特征列表 [M][B*V, D]
        labels: 真实标签 [B]
        num_views: 视图数量 (1或2)
        """
        # 特征投影
        stu_proj = F.normalize(self.projection(stu_features), dim=-1)
        tea_projections = [F.normalize(self.projection(tf), dim=-1) for tf in tea_features_list]

        # 学生内部对比（仅当视图数>1时）
        intra_loss = 0.0
        if num_views > 1:
            intra_loss = self._supervised_contrastive(
                features=stu_proj, 
                labels=labels.repeat(num_views),
                num_views=num_views
            )

        # 学生-教师对比
        inter_loss = 0.0
        for tea_proj in tea_projections:
            inter_loss += self._supervised_contrastive(
                features=torch.cat([stu_proj, tea_proj], dim=0),
                labels=torch.cat([labels.repeat(num_views), labels.repeat(num_views)]),
                num_views=2
            )
        
        return intra_loss + inter_loss / len(tea_projections)

    def _supervised_contrastive(self, features, labels, num_views):
        """核心监督对比损失计算"""
        device = features.device
        batch_size = len(labels) // num_views
        
        # 构建标签矩阵
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # 相似度矩阵
        similarity = torch.matmul(features, features.T) / self.temperature
        
        # 排除对角线
        logits_mask = torch.ones_like(mask) - torch.eye(len(mask)).to(device)
        mask = mask * logits_mask
        
        # 计算log_prob
        exp_logits = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_logits.sum(dim=1, keepdim=True))
        
        # 计算每个正样本对的平均log概率
        per_instance_loss = - (mask * log_prob).sum(1) / mask.sum(1)
        
        # 按视图数调整损失计算
        if num_views > 1:
            per_instance_loss = per_instance_loss.view(batch_size, num_views).mean(dim=1)
        
        return per_instance_loss.mean()

class CFLWithSupervisedContrast(nn.Module):
    def __init__(self, 
                 stu_dim=2048, 
                 tea_dims=[2048, 2048], 
                 alpha=1.0,
                 beta=0.5,
                 gamma=0.1,
                 temp=0.1):
        super().__init__()
        self.alpha = alpha  # MMD权重
        self.beta = beta    # MSE权重
        self.gamma = gamma  # 对比损失权重
        
        # 初始化各组件
        # self.cfl_blk = CFL_ConvBlock(stu_dim, tea_dims, 128)
        self.contrast_module = SupervisedContrastiveCFL(temperature=temp)
        
    def forward(self, fs, ft_list, labels, num_views=2):
        """
        fs: 学生特征 [B*V, D]
        ft_list: 教师特征列表 [M][B*V, D]
        labels: 真实标签 [B]
        num_views: 视图数量
        """
        # CFL基础处理
        # (hs, ht), (ft_proj, ft) = self.cfl_blk(fs, ft_list)
        
        # 计算MMD
        # mmd_loss = sum(calc_mmd(hs, h_t) for h_t in ht)
        
        # 计算MSE
        # mse_loss = sum(F.mse_loss(p, t) for p, t in zip(ft_proj, ft))
        
        # 监督对比损失
        contrast_loss = self.contrast_module(
            stu_features=fs,
            tea_features_list=ft_list,
            labels=labels,
            num_views=num_views
        )
        
        # # 综合损失
        # total_loss = (
        #     self.alpha * mmd_loss +
        #     self.beta * mse_loss +
        #     self.gamma * contrast_loss
        # )
        
        return {
            # 'total_loss': total_loss,
            # 'mmd_loss': mmd_loss,
            # 'mse_loss': mse_loss,
            'contrast_loss': contrast_loss
        }

def calc_mmd(x, y):
    """简化版MMD计算"""
    xx = torch.mean(x @ x.t())
    yy = torch.mean(y @ y.t())
    xy = torch.mean(x @ y.t())
    return xx + yy - 2*xy


# # 初始化
# model = CFLWithSupervisedContrast(
#     stu_dim=2048,
#     tea_dims=[2048, 2048],  # 两个教师的不同维度
#     gamma=0.5
# )

# 双视图输入
# student_feats = torch.randn(128, 2048)  # 64样本x2视图
# teacher_feats = [
#     torch.randn(128, 1024),
#     torch.randn(128, 2048)
# ]
# labels = torch.randint(0, 100, (64,))

# loss_dict = model(
#     fs=student_feats,
#     ft_list=teacher_feats,
#     labels=labels,
#     num_views=2
# )

# # 单视图输入
# student_feats = torch.randn(64, 128, 7, 7)
# teacher_feats = [
#     torch.randn(64, 128, 7, 7),
#     torch.randn(64, 128, 7, 7)
# ]
# labels = torch.randint(0, 100, (64,))
# # loss_dict = model(
# #     fs=student_feats,
# #     ft_list=teacher_feats,
# #     labels=labels,
# #     num_views=1
# # )

# # print(loss_dict)

# # ContrastiveCFLoss = ContrastiveCFLoss()
# # loss = ContrastiveCFLoss(student_feats, teacher_feats)
# # print('###', loss)

# Sup_ContrastiveCFLoss = Sup_ContrastiveCFLoss()
# loss = Sup_ContrastiveCFLoss(student_feats, teacher_feats, labels)
# print('^^^', loss)
