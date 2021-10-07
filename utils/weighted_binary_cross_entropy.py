import torch
import torch.nn as nn
import torch.nn.functional as F

def weighted_binary_cross_entropy(sigmoid_x, targets, pos_weight, neg_weight, weight=None, size_average=True, reduce=True):
    """
    Args:
        sigmoid_x: predicted probability of size [N,C], N sample and C Class. Eg. Must be in range of [0,1], i.e. Output from Sigmoid.
        targets: true value, one-hot-like vector of size [N,C]
        pos_weight: Weight for postive sample
    """
    if not (targets.size() == sigmoid_x.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(targets.size(), sigmoid_x.size()))

    sigmoid_x = sigmoid_x.clamp(min=1e-7)
    loss = - pos_weight * targets * sigmoid_x.log().clamp(min=-100) - neg_weight * (1 - targets) * (1 - sigmoid_x).log().clamp(min=-100)

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=torch.tensor(1.), neg_weight=torch.tensor(1.), weight=None, PosNegWeightIsDynamic=False, WeightIsDynamic=False, size_average=True, reduce=True):
        """
        Args:
            pos_weight = Weight for postive samples. Size [1,C]
            weight = Weight for Each class. Size [1,C]
            PosWeightIsDynamic: If True, the pos_weight is computed on each batch. If pos_weight is None, then it remains None.
            WeightIsDynamic: If True, the weight is computed on each batch. If weight is None, then it remains None.
        """
        super().__init__()

        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)
        self.register_buffer('neg_weight', neg_weight)
        self.size_average = size_average
        self.reduce = reduce
        self.PosNegWeightIsDynamic = PosNegWeightIsDynamic

    def forward(self, input, target):
        if self.PosNegWeightIsDynamic:
            len_target = target.numel()
            positive_counts = target.sum()
            negative_counts = len_target - positive_counts
            self.pos_weight = len_target / positive_counts.clamp(min = 1)
            self.neg_weight = len_target / negative_counts.clamp(min = 1)

        if self.weight is not None:
            return weighted_binary_cross_entropy(input, target,
                                                 self.pos_weight,
                                                 self.neg_weight,
                                                 weight=self.weight,
                                                 size_average=self.size_average,
                                                 reduce=self.reduce)
        else:
            return weighted_binary_cross_entropy(input, target,
                                                 self.pos_weight,
                                                 self.neg_weight,
                                                 weight=None,
                                                 size_average=self.size_average,
                                                 reduce=self.reduce)