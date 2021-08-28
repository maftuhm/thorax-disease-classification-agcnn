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
    def __init__(self, backbone = 'resnet50', num_classes = 14, add_layer = False, **kwargs):

        super(FusionNet, self).__init__()

        # ----------------- Backbone -----------------

        self.add_layer = add_layer

        if backbone == 'resnet50':
            len_input = 2048
        elif backbone == 'densenet121':
            len_input = 1024
        else:
            raise Exception("backbone must be resnet50 or densenet121")

        if add_layer:
            self.fc = nn.Linear(len_input * 2, len_input // 2)
            self.fc1 = nn.Linear(len_input // 2, len_input // 4)
            self.fc2 = nn.Linear(len_input // 4, len_input // 8)
            self.fc3 = nn.Linear(len_input // 8, num_classes)
            self.relu = nn.ReLU()
        else:
            self.fc = nn.Linear(len_input * 2, num_classes)

        self.sigmoid = nn.Sigmoid()

    def forward(self, pool):
        out = self.fc(pool)

        if self.add_layer:
            out = self.relu(out)
            out = self.fc1(out)
            out = self.relu(out)
            out = self.fc2(out)
            out = self.relu(out)
            out = self.fc3(out)

        # out = self.sigmoid(out)

        return {'out': out}