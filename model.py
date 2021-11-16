import torch
import torch.nn as nn
import torch.nn.functional as F
from models import ResNet50, ResNet101, DenseNet121
from utils import AttentionMaskInference, LSEPool2d

def MainNet(backbone = 'resnet50', **kwargs):

    # ----------------- Backbone -----------------
    if backbone == 'resnet50':
        model = ResNet50(**kwargs)
    elif backbone == 'resnet101':
        model = ResNet101(**kwargs)
    elif backbone == 'densenet121':
        model = DenseNet121(**kwargs)
    else:
        raise Exception("backbone must be resnet50, resnet101 or densenet121")

    print(" Backbone \t\t:", model.__class__.__name__)
    print(" Pretrained model \t:", kwargs.get('pretrained', True))
    print(" Last pooling layer\t:", model.last_pool.__class__.__name__)
    if isinstance(model.last_pool, LSEPool2d):
        print(" lse pooling controller :", model.last_pool.controller)

    return model


class FusionNet(nn.Module):
    def __init__(self, **kwargs):

        super(FusionNet, self).__init__()

        # ----------------- Backbone -----------------

        backbone = kwargs.get('backbone', 'resnet50')
        num_classes = kwargs.get('num_classes', 14)
        self.add_layer = kwargs.get('add_layer', False)

        if backbone == 'resnet50':
            len_input = 2048
        elif backbone == 'densenet121':
            len_input = 1024
        else:
            raise Exception("backbone must be resnet50 or densenet121")

        if self.add_layer:
            self.fc = nn.Linear(len_input * 2, len_input)
            self.fc1 = nn.Linear(len_input, len_input // 2)
            self.fc2 = nn.Linear(len_input // 2, len_input // 4)
            self.fc3 = nn.Linear(len_input // 4, len_input // 8)
            self.fc4 = nn.Linear(len_input // 8, len_input // 16)
            self.fc5 = nn.Linear(len_input // 16, num_classes)
            self.relu = nn.ReLU()
        else:
            self.fc = nn.Linear(len_input * 2, num_classes)

    def forward(self, pool):
        out = self.fc(pool)

        if self.add_layer:
            out = self.relu(out)
            out = self.fc1(out)
            out = self.fc2(out)
            out = self.fc3(out)
            out = self.fc4(out)
            out = self.fc5(out)

        out = torch.sigmoid(out)

        result = {
            'score': out
        }

        return result

    def load_branch_weight(self, global_weight, local_weight):
        self.global_net.load_state_dict(global_weight)
        self.local_net.load_state_dict(local_weight)
        print(" Global and Local model weight have been loaded.")
        self.reconstruct_branch()

    def reconstruct_branch(self):
        global_net = list(self.global_net.children())
        self.global_features = nn.Sequential(*global_net[:-2])
        self.global_pool = global_net[-2]

        local_net = list(self.local_net.children())
        self.local_pool = nn.Sequential(*local_net[:-1])

        del self.global_net, self.local_net, global_net, local_net
        torch.cuda.empty_cache()
        print(" Global and Local model have been reconstructed.")

    # def load_state_dict(self, state_dict, strict = True):
    #     self.reconstruct_branch()
    #     return super().load_state_dict(state_dict, strict=strict)