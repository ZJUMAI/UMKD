import numpy as np 
import torch.nn.functional as F

class AverageMeter(object):
    """Average Meter"""
    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()
    
    def reset(self, key=None):
        if key is None:
            self.reset_all()
            return
        item = self.book.get(key, None)
        if item is not None:
            item[0] = 0 # value 
            item[1] = 0 # count
    
    def update(self, key, val):
        record = self.book.get(key, None)
        if record is None:
            self.book[key] = [val, 1]
        else:
            record[0]+=val
            record[1]+=1

    def get_results(self, key):
        record = self.book.get(key, None)
        assert record is not None
        return record[0] / record[1]


class StreamClsMetrics(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        #混淆矩阵，记录模型预测的类别与真实标签之间的关系，行是真实，列是预测标签
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.mae = 0

    def update(self, pred, target):
        self.mae = MAE(pred, target)
        pred = pred.max(dim=1)[1].cpu().numpy().astype(np.uint8)#对于每个样本，返回每个样本的最大值和对应的索引，[1] 是选择索引，即预测的类别。
        target = target.cpu().numpy().astype(np.uint8)
        for lt, lp in zip(target, pred):#组合成配对，分别表示每个样本的真实标签和预测标签。
            self.confusion_matrix[lt][lp] += 1
            #根据真实标签 lt 和预测标签 lp 更新混淆矩阵。相应的混淆矩阵位置加 1。
    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k != "Confusion Matrix" and k != "Class IoU":
                string += "%s: %f\n" % (k, v)
        return string

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) +
                              hist.sum(axis=0) - np.diag(hist))
########均值交并比
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
########加权准确率
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
########每个类别的IoU
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {
            "Confusion Matrix": hist,
            "Overall Acc": acc,
            "MAE": self.mae,
            "Mean Acc": acc_cls,
            "FreqW Acc": fwavacc,
            "Mean IoU": mean_iu,
            "Class IoU": cls_iu,
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.mae = 0

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
    #每个类别都有一个预测概率，乘以对应的类别，得到最后的期望标签
    mae = sum(abs(exp_data - target_data)) * 1.0 / len(target_data)

    return mae
