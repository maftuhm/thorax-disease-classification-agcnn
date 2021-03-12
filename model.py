import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.model = model
        
        if self.model == 'resnet50':
            self.backbone = torchvision.models.resnet50(pretrained = True)
            self.removed = list(self.backbone.children())[:-2]
            self.fc = nn.Linear(2048, 14)

        if self.model == 'densenet121':
            self.backbone = torchvision.models.densenet121(pretrained = True)
            self.removed = list(self.backbone.children())[:-1]
            self.fc = nn.Linear(1048, 14)
            
        self.layers = nn.Sequential(*self.removed)
        self.pool = nn.MaxPool2d(kernel_size = 7, stride = 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        features = self.layers(x)
        pool = self.pool(features)
        pool = pool.view(pool.size(0), -1)
        x = self.fc(pool)
        x = self.sigmoid(x)
        
        return x, features, pool

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