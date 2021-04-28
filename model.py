import torch
import torch.nn as nn
import torch.nn.functional as F
from models import ResNet50, DenseNet121

class ResAttCheXNet(nn.Module):
    def __init__(self, last_pool = 'lse', lse_pool_controller = 5, backbone = 'resnet50', pretrained = True,
                num_classes = 14, criterion = 'WeightedBCELoss', DynamicWeightLoss = True, **kwargs):
        super(ResAttCheXNet, self).__init__()

        # ----------------- Backbone -----------------
        if backbone == 'resnet50':
            self.backbone = ResNet50(pretrained = pretrained,
                                    num_classes = num_classes,
                                    last_pool = last_pool,
                                    lse_pool_controller = lse_pool_controller)

        elif backbone == 'densenet121':
            self.backbone = DenseNet121(pretrained = pretrained,
                                        num_classes = num_classes,
                                        last_pool = last_pool,
                                        lse_pool_controller = lse_pool_controller)
        else:
            raise Exception("backbone must be resnet50 or densenet121")

        # ----------------- Loss Function -----------------
        if criterion == 'WeightedBCELoss':
            self.criterion = WeightedBCELoss(PosNegWeightIsDynamic = DynamicWeightLoss)
        elif criterion == 'BCELoss':
            self.criterion = nn.BCELoss()
        else:
            raise Exception("Loss function must be BCELoss or WeightedBCELoss")

    def forward(self, image, label = None):
        out, features, pool = self.backbone(image)

        # residual_features = self.discriminative_features(features, self.attention(features))
        # scores_residual = self.fc_residual(torch.flatten(residual_features, 1))

        if label is not None:
            loss = self.criterion(out, label)
            # loss_residual = self.criterion(scores_residual, label)
        else:
            loss = torch.tensor(0, dtype=image.dtype, device=image.device)
            # loss_residual = torch.tensor(0, dtype=image.dtype, device=image.device)

        output = {
            'scores' : out,
            # 'scores_residual' : scores_residual,
            'features' : features,
            'pool' : pool,
            'loss' : loss,
            # 'loss_residual' : loss_residual
        }

        return output


class FusionNet(nn.Module):
    def __init__(self, backbone = 'resnet50', num_classes = 14,
                criterion = 'WeightedBCELoss', DynamicWeightLoss = True, **kwargs):

        super(FusionNet, self).__init__()

        # ----------------- Loss Function -----------------
        if criterion == 'WeightedBCELoss':
            self.criterion = WeightedBCELoss(PosNegWeightIsDynamic = DynamicWeightLoss)
        elif criterion == 'BCELoss':
            self.criterion = nn.BCELoss()
        else:
            raise Exception("Loss function must be BCELoss or WeightedBCELoss")

        # ----------------- Backbone -----------------
        if backbone == 'resnet50':
            self.fc = nn.Linear(2048 * 2, num_classes)
        elif backbone == 'densenet121':
            self.fc = nn.Linear(1024 * 2, num_classes)
        else:
            raise Exception("backbone must be resnet50 or densenet121")

        self.sigmoid = nn.Sigmoid()

    def forward(self, pool, label = None):
        scores = self.fc(pool)
        scores = self.sigmoid(scores)

        if label is not None:
            loss = self.criterion(scores, label)
        else:
            loss = torch.tensor(0, dtype=pool.dtype, device=pool.device)

        output = {
            'scores' : scores,
            'loss' : loss
        }

        return output

class LSEPool2d(nn.Module):
    def __init__(self, r = 10):
        super(LSEPool2d, self).__init__()
        self.r = r
        self.maxpool = nn.MaxPool2d(kernel_size = 7, stride = 1)

    def forward(self, x):
        xmaxpool = torch.abs(x)
        xmaxpool = self.maxpool(xmaxpool)
        xpool = (1 / (x.shape[-1] * x.shape[-2])) * torch.sum(torch.exp(self.r * (x - xmaxpool)), dim = (-2, -1))
        xpool  = xmaxpool + (1 / self.r) * torch.log(xpool).unsqueeze(-1).unsqueeze(-1)
        return xpool

def weighted_binary_cross_entropy(sigmoid_x, targets, pos_weight, neg_weight, weight=None, size_average=True, reduce=True):
    """
    Args:
        sigmoid_x: predicted probability of size [N,C], N sample and C Class. Eg. Must be in range of [0,1], i.e. Output from Sigmoid.
        targets: true value, one-hot-like vector of size [N,C]
        pos_weight: Weight for postive sample
    """
    if not (targets.size() == sigmoid_x.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(targets.size(), sigmoid_x.size()))

    sigmoid_x = sigmoid_x.clamp(min=1e-7, max=1-1e-7)
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
    def __init__(self, pos_weight=1, neg_weight=1, weight=None, PosNegWeightIsDynamic=False, WeightIsDynamic=False, size_average=True, reduce=True):
        """
        Args:
            pos_weight = Weight for postive samples. Size [1,C]
            weight = Weight for Each class. Size [1,C]
            PosWeightIsDynamic: If True, the pos_weight is computed on each batch. If pos_weight is None, then it remains None.
            WeightIsDynamic: If True, the weight is computed on each batch. If weight is None, then it remains None.
        """
        super().__init__()

        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', torch.tensor(pos_weight))
        self.register_buffer('neg_weight', torch.tensor(neg_weight))
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