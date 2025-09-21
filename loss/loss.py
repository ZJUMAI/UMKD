import torch
import torch.nn as nn
import torch.nn.functional as F
from .mmd import calc_mmd
from typing import List, Optional, Union

def euclidean_loss(hs, ht):
    euc_loss = 0.0
    for ht_i in ht:
        euc_loss += euclidean(hs, ht_i)
    return euc_loss

def euclidean(a, b):
    """
    计算两个张量 a 和 b 之间的欧氏距离损失。
    
    参数:
    a (Tensor): 输入的第一个张量，形状为 (N, C, H, W)
    b (Tensor): 输入的第二个张量，形状为 (N, C, H, W)
    
    返回:
    Tensor: 两个张量之间的欧氏距离损失
    """
    # 计算差异
    diff = a - b
    # 计算平方差
    squared_diff = diff ** 2
    # 对所有元素求和
    sum_squared_diff = squared_diff.sum()
    # 返回欧氏距离
    loss = torch.sqrt(sum_squared_diff)
    return loss


def cosine_similarity_loss(hs, ht):
    cos_loss = 0.0
    for ht_i in ht:
        cos_loss += cosine_similarity(hs, ht_i)
    return cos_loss

def cosine_similarity(a, b):
    """
    计算两个张量 a 和 b 之间的余弦相似度损失。
    
    参数:
    a (Tensor): 输入的第一个张量，形状为 (N, C, H, W)
    b (Tensor): 输入的第二个张量，形状为 (N, C, H, W)
    
    返回:
    Tensor: 两个张量之间的余弦相似度损失
    """
    # 将 a 和 b 展平为二维张量，形状变为 (N, C*H*W)
    a_flat = a.view(a.size(0), -1)
    b_flat = b.view(b.size(0), -1)

    # 计算余弦相似度
    cosine_sim = F.cosine_similarity(a_flat, b_flat, dim=1)

    # 计算余弦相似度损失，1 - cosine similarity
    loss = 1 - cosine_sim.mean()  # 平均余弦相似度损失
    return loss


class CFLoss(nn.Module):
    """ Common Feature Learning Loss
        CF Loss = MMD + beta * MSE
MMD损失: 最大均值差异损失用于对齐学生和教师模型的特征分布，减少学生和教师模型之间的分布差异。
MSE损失: 重构损失，保证教师模型的特征能够重构回去。
    """
    # sigmas：这是一个超参数列表，用于设置 MMD 损失中的核函数的标准差。
    # 小的 sigma 值意味着核函数更加局部化，更加敏感。
    # 大的 sigma 值意味着核函数更加全局化，捕捉到更广泛的模式
    def __init__(self, sigmas=[0.001, 0.01, 0.05, 0.1, 0.2, 1, 2], normalized=True, beta=1.0):
        super(CFLoss, self).__init__()
        self.sigmas = sigmas
        self.normalized = normalized
        self.beta = beta

    def forward(self, hs, ht, ft_, ft):
        mmd_loss = 0.0
        mse_loss = 0.0
        for ht_i in ht:
            mmd_loss += calc_mmd(hs, ht_i, sigmas=self.sigmas, normalized=self.normalized)
        for i in range(len(ft_)):
            mse_loss += F.mse_loss(ft_[i], ft[i])
        
        return mmd_loss + self.beta*mse_loss

class CFLoss_SA(nn.Module):
    """ Common Feature Learning Loss
        CF Loss = MMD + beta * MSE
MMD损失: 最大均值差异损失用于对齐学生和教师模型的特征分布，减少学生和教师模型之间的分布差异。
MSE损失: 重构损失，保证教师模型的特征能够重构回去。
    """
    # sigmas：这是一个超参数列表，用于设置 MMD 损失中的核函数的标准差。
    # 小的 sigma 值意味着核函数更加局部化，更加敏感。
    # 大的 sigma 值意味着核函数更加全局化，捕捉到更广泛的模式
    def __init__(self, sigmas=[0.001, 0.01, 0.05, 0.1, 0.2, 1, 2], normalized=True, beta=1.0):
        super(CFLoss_SA, self).__init__()
        self.sigmas = sigmas
        self.normalized = normalized
        self.beta = beta

    def forward(self, hs, ht):
        mmd_loss = 0.0
        # mse_loss = 0.0
        for ht_i in ht:
            mmd_loss += calc_mmd(hs, ht_i, sigmas=self.sigmas, normalized=self.normalized)
        # for i in range(len(ft_)):
        #     mse_loss += F.mse_loss(ft_[i], ft[i])

        return mmd_loss
    
class CFLoss_UC(nn.Module):
    """ Common Feature Learning Loss
        CF Loss = MMD + beta * MSE
MMD损失: 最大均值差异损失用于对齐学生和教师模型的特征分布，减少学生和教师模型之间的分布差异。
MSE损失: 重构损失，保证教师模型的特征能够重构回去。
    """
    # sigmas：这是一个超参数列表，用于设置 MMD 损失中的核函数的标准差。
    # 小的 sigma 值意味着核函数更加局部化，更加敏感。
    # 大的 sigma 值意味着核函数更加全局化，捕捉到更广泛的模式
    def __init__(self, sigmas=[0.001, 0.01, 0.05, 0.1, 0.2, 1, 2], normalized=True, beta=1.0):
        super(CFLoss_UC, self).__init__()
        self.sigmas = sigmas
        self.normalized = normalized
        self.beta = beta

    def forward(self, hs, ht, ft_, ft, uc1, uc2):
        mmd_loss = 0.0
        mse_loss = 0.0
        for idx, ht_i in enumerate(ht):
            if idx == 0:
                mmd_loss += calc_mmd(hs, ht_i, sigmas=self.sigmas, normalized=self.normalized) * (1-uc1)
            elif idx == 1:
                mmd_loss += calc_mmd(hs, ht_i, sigmas=self.sigmas, normalized=self.normalized) * (1-uc2)
        for i in range(len(ft_)):
            mse_loss += F.mse_loss(ft_[i], ft[i])
        
        return mmd_loss + self.beta*mse_loss
    
#软目标交叉熵损失
class SoftCELoss(nn.Module):
    """ KD Loss Function (CrossEntroy for soft targets)
    """
    # T是温度参数，用于对目标分布进行软化。
    # 温度值越高，目标分布的概率分布越平滑，从而使得模型更容易学习教师模型的知识。
    # alpha是硬标签（即标准的分类标签）对总损失的贡献权重。
    # alpha 控制了硬标签与软标签之间的平衡。如果 alpha = 0，则只使用软标签；
    # 如果 alpha = 1，则硬标签和软标签的贡献相等。
    def __init__(self, T=1.0, alpha=1.0):
        super(SoftCELoss, self).__init__()
        self.T = T
        self.alpha = alpha

    def forward(self, logits, targets, hard_targets=None):
        ce_loss = soft_cross_entropy(logits, targets, T=self.T)
#########为了解决类别不平衡问题，这里使用bceloss来计算损失#############
        #ce_loss = soft_binary_cross_entropy(logits, targets, T=self.T)
        if hard_targets is not None and self.alpha != 0.0:
            ce_loss += self.alpha*F.cross_entropy(logits, hard_targets)
        return ce_loss

def soft_cross_entropy(logits, target, T=1.0, size_average=True, target_is_prob=False):
    """ Cross Entropy for soft targets
    
    **Parameters:**
        - **logits** (Tensor): logits score (e.g. outputs of fc layer)
        - **targets** (Tensor): logits of soft targets
        - **T** (float): temperature　of distill
        - **size_average**: average the outputs
        - **target_is_prob**: set True if target is already a probability.
    """
    if target_is_prob:
        p_target = target
    else:
        p_target = F.softmax(target/T, dim=1)
    
    logp_pred = F.log_softmax(logits/T, dim=1)
    # F.kl_div(logp_pred, p_target, reduction='batchmean')*T*T
#####计算交叉熵损失
    ce = torch.sum(-p_target * logp_pred, dim=1)
    if size_average: # 是否对所有样本的损失进行平均
        return ce.mean() * T * T
    else:
        return ce * T * T

def soft_binary_cross_entropy(logits, target, T=1.0, size_average=True, target_is_prob=False):
    if target_is_prob:
        p_target = target
    else:
        p_target = torch.sigmoid(target / T)
    p_pred = torch.sigmoid(logits / T)
    # Use BCELoss to calculate the loss
    #'mean'（默认值）:含义：返回所有样本损失的均值。
    #'none':含义：返回每个样本的损失，而不是对所有样本的损失进行求和或求平均。
    #'sum':含义：返回所有样本损失的总和。
    criterion = nn.BCELoss(reduction='none') 
    # Calculate binary cross-entropy loss
    bce = criterion(p_pred, p_target)
    if size_average:
        return bce.mean() * T * T  # Scale the loss by T^2
    else:
        return bce * T * T  # Return unscaled loss

class FocalLoss(nn.Module):
    def __init__(self, alpha: Union[List[float], float],
                 gamma: Optional[int] = 2,
                 with_logits: Optional[bool] = True):
        """

        :param alpha: 每个类别的权重
        :param gamma:
        :param with_logits: 是否经过softmax或者sigmoid
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = torch.FloatTensor([alpha]) if isinstance(alpha, float) else torch.FloatTensor(alpha)
        self.smooth = 1e-8
        self.with_logits = with_logits

    def _binary_class(self, input, target):
        prob = torch.sigmoid(input) if self.with_logits else input
        prob += self.smooth
        alpha = self.alpha.to(target.device)
        loss = -alpha * torch.pow(torch.sub(1.0, prob), self.gamma) * torch.log(prob)
        return loss

    def _multiple_class(self, input, target):
        prob = F.softmax(input, dim=1) if self.with_logits else input

        alpha = self.alpha.to(target.device)
        alpha = alpha.gather(0, target)

        target = target.view(-1, 1)

        prob = prob.gather(1, target).view(-1) + self.smooth  # avoid nan
        logpt = torch.log(prob)

        loss = -alpha * torch.pow(torch.sub(1.0, prob), self.gamma) * logpt
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """

        :param input: 维度为[bs, num_classes]
        :param target: 维度为[bs]
        :return:
        """
        if len(input.shape) > 1 and input.shape[-1] != 1:
            loss = self._multiple_class(input, target)
        else:
            loss = self._binary_class(input, target)

        return loss.mean()


class DiceLoss(nn.Module):
    """
    Dice coefficient for short, is an F1-oriented statistic used to gauge the similarity of two sets.

    Args:
        smooth (float, optional): a manual smooth value for numerator and denominator.
        square_denominator (bool, optional): [True, False], specifies whether to square the denominator in the loss function.
        with_logits (bool, optional): [True, False], specifies whether the input tensor is normalized by Sigmoid/Softmax funcs.
        ohem_ratio: max ratio of positive/negative, defautls to 0.0, which means no ohem.
        alpha: dsc alpha
    """

    def __init__(self,
                 smooth: Optional[float] = 1e-4,  # 对应公式中的$$\gamma$$
                 square_denominator: Optional[bool] = False,
                 with_logits: Optional[bool] = True,
                 ohem_ratio: Optional[float] = 0.0,  # 正负样本的最大比例，超过这个比例的负样本则不计算loss
                 alpha: Optional[float] = 0.0,
                 reduction: Optional[str] = "mean",
                 index_label_position: Optional[bool] = True,
                 set_level: Optional[bool] = True  # dice对应set-level or individual
                 ) -> None:
        super(DiceLoss, self).__init__()

        self.reduction = reduction
        self.with_logits = with_logits
        self.smooth = smooth
        self.square_denominator = square_denominator
        self.ohem_ratio = ohem_ratio
        self.alpha = alpha
        self.index_label_position = index_label_position
        self.set_level = set_level

    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits_size = input.shape[-1]

        if len(input.shape) > 1 and logits_size != 1:
            loss = self._multiple_class(input, target, logits_size, mask=mask)
        else:
            loss = self._binary_class(input, target, mask=mask)

        # reduction仅对`set_level=False`生效
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    # TODO 注释为原论文代码，存在问题
    # 1. 二分类仅实现set-level dice coefficient
    # 2. 多分类实现的是individual dice coefficient，但`square_denominator=False`时，又出现set-level计算: `flat_input.sum()`
    # def _compute_dice_loss(self, flat_input, flat_target):
    #     flat_input = ((1 - flat_input) ** self.alpha) * flat_input
    #     interection = torch.sum(flat_input * flat_target, -1)
    #     if not self.square_denominator:
    #         loss = 1 - ((2 * interection + self.smooth) /
    #                     (flat_input.sum() + flat_target.sum() + self.smooth))
    #     else:
    #         loss = 1 - ((2 * interection + self.smooth) /
    #                     (torch.sum(torch.square(flat_input, ), -1) + torch.sum(torch.square(flat_target), -1) + self.smooth))
    #
    #     return loss

    def _compute_dice_loss(self, flat_input, flat_target):
        """
        二分类增加individual dice coefficient的实现
        多分类增加set-level dice coefficient的实现，并统一计算维度
        :param flat_input:
        :param flat_target:
        :return:
        """
        if self.set_level:
            flat_input = flat_input.view(-1)
            flat_target = flat_target.view(-1)
        else:
            flat_input = flat_input.view(-1, 1)
            flat_target = flat_target.view(-1, 1)

        flat_input = ((1 - flat_input) ** self.alpha) * flat_input
        interection = torch.sum(flat_input * flat_target, -1)

        if self.square_denominator:
            flat_input = torch.square(flat_input)
            flat_target = torch.square(flat_target)

        loss = 1 - ((2 * interection + self.smooth) /
                    (torch.sum(flat_input, -1) + torch.sum(flat_target, -1) + self.smooth))

        return loss

    def _multiple_class(self, input, target, logits_size, mask=None):
        flat_input = input
        flat_target = F.one_hot(target,
                                num_classes=logits_size).float() if self.index_label_position else target.float()
        flat_input = torch.nn.Softmax(dim=1)(flat_input) if self.with_logits else flat_input

        if mask is not None:
            mask = mask.float()
            flat_input = flat_input * mask
            flat_target = flat_target * mask
        else:
            mask = torch.ones_like(target)

        loss = None
        if self.ohem_ratio > 0:
            mask_neg = torch.logical_not(mask)
            for label_idx in range(logits_size):
                pos_example = target == label_idx
                neg_example = target != label_idx

                pos_num = pos_example.sum()
                neg_num = mask.sum() - (pos_num - (mask_neg & pos_example).sum())
                keep_num = min(int(pos_num * self.ohem_ratio / logits_size), neg_num)

                if keep_num > 0:
                    neg_scores = torch.masked_select(flat_input, neg_example.view(-1, 1).bool()).view(-1, logits_size)
                    neg_scores_idx = neg_scores[:, label_idx]
                    neg_scores_sort, _ = torch.sort(neg_scores_idx, )
                    threshold = neg_scores_sort[-keep_num + 1]
                    cond = (torch.argmax(flat_input, dim=1) == label_idx & flat_input[:,
                                                                           label_idx] >= threshold) | pos_example.view(
                        -1)
                    ohem_mask_idx = torch.where(cond, 1, 0)

                    flat_input_idx = flat_input[:, label_idx]
                    flat_target_idx = flat_target[:, label_idx]

                    flat_input_idx = flat_input_idx * ohem_mask_idx
                    flat_target_idx = flat_target_idx * ohem_mask_idx
                else:
                    flat_input_idx = flat_input[:, label_idx]
                    flat_target_idx = flat_target[:, label_idx]

                loss_idx = self._compute_dice_loss(flat_input_idx.view(-1, 1), flat_target_idx.view(-1, 1))
                if loss is None:
                    loss = loss_idx
                else:
                    loss += loss_idx
            return loss

        else:
            for label_idx in range(logits_size):
                pos_example = target == label_idx
                flat_input_idx = flat_input[:, label_idx]
                flat_target_idx = flat_target[:, label_idx]

                loss_idx = self._compute_dice_loss(flat_input_idx.view(-1, 1), flat_target_idx.view(-1, 1))
                if loss is None:
                    loss = loss_idx
                else:
                    loss += loss_idx
            return loss

    def _binary_class(self, input, target, mask=None):
        flat_input = input.view(-1)
        flat_target = target.view(-1).float()
        flat_input = torch.sigmoid(flat_input) if self.with_logits else flat_input

        if mask is not None:
            mask = mask.float()
            flat_input = flat_input * mask
            flat_target = flat_target * mask
        else:
            mask = torch.ones_like(target)

        if self.ohem_ratio > 0:
            pos_example = target > 0.5
            neg_example = target <= 0.5
            mask_neg_num = mask <= 0.5

            pos_num = pos_example.sum() - (pos_example & mask_neg_num).sum()
            neg_num = neg_example.sum()
            keep_num = min(int(pos_num * self.ohem_ratio), neg_num)

            neg_scores = torch.masked_select(flat_input, neg_example.bool())
            neg_scores_sort, _ = torch.sort(neg_scores, )
            threshold = neg_scores_sort[-keep_num + 1]
            cond = (flat_input > threshold) | pos_example.view(-1)
            ohem_mask = torch.where(cond, 1, 0)
            flat_input = flat_input * ohem_mask
            flat_target = flat_target * ohem_mask

        return self._compute_dice_loss(flat_input, flat_target)

    def __str__(self):
        return f"Dice Loss smooth:{self.smooth}, ohem: {self.ohem_ratio}, alpha: {self.alpha}"

    def __repr__(self):
        return str(self)

if __name__ == '__main__':
    # for test
    import numpy as np
    np.random.seed(2022)

    multi_pred, multi_target = np.random.random([32, 3]), np.random.randint(0, 3, [32])
    binary_pred, binary_target = np.random.random([32]), np.random.randint(0, 2, [32])

    focal = FocalLoss([0.25, 0.25, 0.5])
    print('*'*20, 'Focal multi class', '*'*20)
    print(focal(torch.FloatTensor(multi_pred), torch.LongTensor(multi_target)))
    focal = FocalLoss(0.25)
    print('*'*20, 'Focal binary class', '*'*20)
    print(focal(torch.FloatTensor(binary_pred), torch.LongTensor(binary_target)))

    dice = DiceLoss(square_denominator=True, set_level=True)
    print('*'*20, 'Dice multi class', '*'*20)
    print(dice(torch.FloatTensor(multi_pred), torch.LongTensor(multi_target)))
    print('*'*20, 'Dice binary class', '*'*20)
    print(dice(torch.FloatTensor(binary_pred), torch.LongTensor(binary_target)))
