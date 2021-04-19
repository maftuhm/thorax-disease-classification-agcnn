import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Net(nn.Module):
    def __init__(self, model_name):
        super(Net, self).__init__()
        self.model_name = model_name
        
        if self.model_name == 'resnet50':
            backbone = models.resnet50(pretrained = True)
            self.features = nn.Sequential(*list(backbone.children())[:-2])
            num_features = backbone.fc.in_features

        if self.model_name == 'densenet121':
            backbone = models.densenet121(pretrained = True)
            self.features = nn.Sequential(*list(backbone.children())[:-1])
            num_features = backbone.classifier.in_features

    
        self.pool = LSEPool2d()
        self.fc = nn.Sequential(
            nn.Linear(num_features, 14),
            nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        pool = self.pool(features)
        out_after_pooling = pool.view(pool.size(0), -1)
        out = self.fc(out_after_pooling)
        return out, features, out_after_pooling

class FusionNet(nn.Module):
    def __init__(self, model):
        super(FusionNet, self).__init__()
        self.model = model

        if self.model == 'resnet50':
            self.fc = nn.Linear(2048*2, 14)

        if self.model == 'densenet121':
            self.fc = nn.Linear(1024*2, 14)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, global_pool, local_pool):
        fusion = torch.cat((global_pool,local_pool), 1)
        out = self.fc(fusion)
        out = self.sigmoid(out)
        return out

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