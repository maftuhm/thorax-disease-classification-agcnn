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

    
        self.maxpool = nn.MaxPool2d(kernel_size = 7, stride = 1)
        self.fc = nn.Sequential(
            nn.Linear(num_features, 14),
            nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        pool = self.maxpool(features)
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
        x = torch.cat((global_pool, local_pool), dim = 1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x