import torch
import torch.nn as nn
import torch.nn.functional as F
from models import ResNet50, DenseNet121

class ResAttCheXNet(nn.Module):
    def __init__(self, last_pool = 'lse', lse_pool_controller = 5, 
                backbone = 'resnet50', pretrained = True, 
                num_classes = 14, group_norm = False, **kwargs):
        super(ResAttCheXNet, self).__init__()

        # ----------------- Backbone -----------------
        if backbone == 'resnet50':
            self.backbone = ResNet50(pretrained = pretrained,
                                    num_classes = num_classes,
                                    last_pool = last_pool,
                                    lse_pool_controller = lse_pool_controller,
                                    group_norm = group_norm,
                                    **kwargs)

        elif backbone == 'densenet121':
            self.backbone = DenseNet121(pretrained = pretrained,
                                        num_classes = num_classes,
                                        last_pool = last_pool,
                                        lse_pool_controller = lse_pool_controller,
                                        memory_efficient = True,
                                        **kwargs)
        else:
            raise Exception("backbone must be resnet50 or densenet121")

        print(" Backbone \t\t:", backbone)
        print(" Last pooling layer\t:", last_pool)
        print(" Pretrained model \t:", pretrained)
        if last_pool == 'lse':
            print(" lse pooling controller :", lse_pool_controller)
        print(" Group normalization \t:", group_norm)

    def forward(self, image):
        out, features, pool = self.backbone(image)

        output = {
            'out' : out,
            'features' : features,
            'pool' : pool,
        }

        return output


class FusionNet(nn.Module):
    def __init__(self, backbone = 'resnet50', num_classes = 14, **kwargs):

        super(FusionNet, self).__init__()

        # ----------------- Backbone -----------------
        if backbone == 'resnet50':
            self.fc = nn.Linear(2048 * 2, num_classes)
        elif backbone == 'densenet121':
            self.fc = nn.Linear(1024 * 2, num_classes)
        else:
            raise Exception("backbone must be resnet50 or densenet121")

        self.sigmoid = nn.Sigmoid()

    def forward(self, pool):
        out = self.fc(pool)
        out = self.sigmoid(out)
        return {'out': out}