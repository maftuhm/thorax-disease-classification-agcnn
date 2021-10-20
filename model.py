import torch
import torch.nn as nn
import torch.nn.functional as F
from models import ResNet50, ResNet101, DenseNet121
from utils import AttentionMaskInference

def MainNet(last_pool = 'lse', lse_pool_controller = 5, 
            backbone = 'resnet50', pretrained = True, 
            num_classes = 14, group_norm = False, **kwargs):

    # ----------------- Backbone -----------------
    if backbone == 'resnet50':
        model = ResNet50(pretrained = pretrained,
                        num_classes = num_classes,
                        last_pool = last_pool,
                        lse_pool_controller = lse_pool_controller,
                        group_norm = group_norm,
                        **kwargs)

    elif backbone == 'resnet101':
        model = ResNet101(pretrained = pretrained,
                        num_classes = num_classes,
                        last_pool = last_pool,
                        lse_pool_controller = lse_pool_controller,
                        **kwargs)

    elif backbone == 'densenet121':
        model = DenseNet121(pretrained = pretrained,
                            num_classes = num_classes,
                            last_pool = last_pool,
                            lse_pool_controller = lse_pool_controller,
                            **kwargs)
    else:
        raise Exception("backbone must be resnet50, resnet101 or densenet121")

    print(" Backbone \t\t:", backbone)
    print(" Last pooling layer\t:", last_pool)
    print(" Pretrained model \t:", pretrained)
    if last_pool == 'lse':
        print(" lse pooling controller :", lse_pool_controller)
        # print(" Group normalization \t:", group_norm)

    return model


class FusionNet(nn.Module):
    def __init__(self, threshold, distance_function, last_pool = 'lse', lse_pool_controller = 5, 
                backbone = 'resnet50', num_classes = 14, group_norm = False, add_layer = False, **kwargs):

        super(FusionNet, self).__init__()

        # ----------------- Backbone -----------------

        self.global_net = MainNet(pretrained = False,
                                backbone = backbone,
                                num_classes = num_classes,
                                last_pool = last_pool,
                                lse_pool_controller = lse_pool_controller,
                                group_norm = group_norm,
                                **kwargs)

        self.local_net = MainNet(pretrained = False,
                                backbone = backbone,
                                num_classes = num_classes,
                                last_pool = last_pool,
                                lse_pool_controller = lse_pool_controller,
                                group_norm = group_norm,
                                **kwargs)

        self.attention_mask = AttentionMaskInference(threshold, distance_function)

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

    def forward(self, img):
        global_features = self.global_features(img)
        global_pool = self.global_pool(global_features)

        out_patches = self.attention_mask(img, global_features.cpu())

        local_pool = self.local_pool(out_patches['image'])

        fusion = torch.cat((global_pool, local_pool), dim = 1).to(img.device)

        out = self.fc(fusion)

        if self.add_layer:
            out = self.relu(out)
            out = self.fc1(out)
            out = self.relu(out)
            out = self.fc2(out)
            out = self.relu(out)
            out = self.fc3(out)

        out = torch.sigmoid(out)

        result = {
            'score': out,
            'patch': out_patches
        }

        return result

    def load_branch_weight(self, global_weight, local_weight):
        self.global_net.load_state_dict(global_weight)
        self.local_net.load_state_dict(local_weight)
        self.reconstruct_branch()

    def reconstruct_branch(self):
        global_net = list(self.global_net.children())
        self.global_features = nn.Sequential(*global_net[:-2])
        self.global_pool = global_net[-2]

        local_net = list(self.local_net.children())
        self.local_pool = nn.Sequential(*local_net[:-1])

        del self.global_net, self.local_net
        torch.cuda.empty_cache()

    def load_state_dict(self, state_dict, strict = True):
        self.reconstruct_branch()
        return super().load_state_dict(state_dict, strict=strict)