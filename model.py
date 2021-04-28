import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResAttCheXNet(nn.Module):
    def __init__(
                self,
                r = 10,
                pooling_layer = 'lse',
                backbone_name = 'resnet50',
                pretrained = True,
                num_classes = 14,
                criterion = 'WeightedBCELoss',
                DynamicWeightLoss = True,
                **kwargs
            ):
        super(ResAttCheXNet, self).__init__()

        self.r = r
        self.pooling_layer = pooling_layer
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.num_classes = num_classes

        self.net_init()

        # ----------------- Loss Function -----------------
        if criterion == 'WeightedBCELoss':
            self.criterion = WeightedBCELoss(PosNegWeightIsDynamic = DynamicWeightLoss)
        elif criterion == 'BCELoss':
            self.criterion = nn.BCELoss()
        else:
            raise Exception("Loss function must be BCELoss or WeightedBCELoss")

    def net_init(self):

        # ----------------- Backbone -----------------
        if self.backbone_name == 'resnet50':
            backbone = models.resnet50(pretrained = self.pretrained)
            self.features = nn.Sequential(*list(backbone.children())[:-2])
            self.num_features = backbone.fc.in_features
        elif self.backbone_name == 'densenet121':
            backbone = models.densenet121(pretrained = self.pretrained)
            self.features = backbone.features
            self.num_features = backbone.classifier.in_features
        else:
            raise Exception("backbone must be resnet50 or densenet121")

        # ----------------- Pooling layer -----------------
        if self.pooling_layer == 'lse':
            self.pool = LSEPool2d(r = self.r)
        elif self.pooling_layer == 'max':
            self.pool = nn.MaxPool2d(kernel_size = 7, stride = 1)
        elif self.pooling_layer == 'avg':
            self.pool = nn.AvgPool2d(kernel_size = 7, stride = 1)
        else:
            raise Exception("pooling layer must be lse, max or avg")

        # ----------------- Full connected layer -----------------
        self.fc = nn.Sequential(
            nn.Linear(self.num_features, self.num_classes),
            nn.Sigmoid()
        )

        # self.fc_residual = nn.Sequential(
        #     nn.Linear(self.num_features * self.num_classes, self.num_classes),
        #     nn.Sigmoid()
        # )

        # ----------------- attention layer -----------------
        # self.attention = nn.Sequential(
        #     nn.Conv2d(self.num_features, self.num_features, kernel_size = 3, padding = 1),
        #     nn.ReLU(),
        #     nn.Conv2d(self.num_features, self.num_features, kernel_size = 3, padding = 1),
        #     nn.ReLU(),
        #     nn.Conv2d(self.num_features, self.num_classes, kernel_size = 1),
        #     nn.Sigmoid()
        # )

    def forward(self, image, label = None):
        features = self.features(image)

        pool = self.pool(features)
        flatten_pool = torch.flatten(pool, 1)
        scores = self.fc(flatten_pool)

        # residual_features = self.discriminative_features(features, self.attention(features))
        # scores_residual = self.fc_residual(torch.flatten(residual_features, 1))

        if label is not None:
            loss = self.criterion(scores, label)
            # loss_residual = self.criterion(scores_residual, label)
        else:
            loss = torch.tensor(0, dtype=image.dtype, device=image.device)
            # loss_residual = torch.tensor(0, dtype=image.dtype, device=image.device)

        output = {
            'scores' : scores,
            # 'scores_residual' : scores_residual,
            'features' : features,
            'pool' : flatten_pool,
            'loss' : loss,
            # 'loss_residual' : loss_residual
        }

        return output

    # def discriminative_features(self, feature, score):
    #     num_class = score.shape[1]
    #     bz, num_channel, h, w = feature.shape

    #     feature_weighted = torch.zeros(bz, num_class, num_channel, dtype = score.dtype, device = score.device)

    #     for i in range(bz):
    #         feature_weighted[i] = (torch.matmul(score[i].unsqueeze(1) + 1, feature[i])).relu().mean(dim = (-2, -1))

    #     return feature_weighted


class FusionNet(nn.Module):
    def __init__(self,
                r = 10,
                backbone_name = 'resnet50',
                num_classes = 14,
                criterion = 'WeightedBCELoss',
                DynamicWeightLoss = True,
                **kwargs):

        super(FusionNet, self).__init__()
        self.r = r
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # ----------------- Loss Function -----------------
        if criterion == 'WeightedBCELoss':
            self.criterion = WeightedBCELoss(PosNegWeightIsDynamic = DynamicWeightLoss)
        elif criterion == 'BCELoss':
            self.criterion = nn.BCELoss()
        else:
            raise Exception("Loss function must be BCELoss or WeightedBCELoss")

        # ----------------- Backbone -----------------
        if self.backbone_name == 'resnet50':
            self.fc = nn.Linear(2048 * 2, self.num_classes)
        elif self.backbone_name == 'densenet121':
            self.fc = nn.Linear(1024 * 2, self.num_classes)
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