import torch
import torch.nn as nn

class LSEPool2d(nn.Module):
    def __init__(self, controller = 10, kernel_size = 7, stride = 1):
        super(LSEPool2d, self).__init__()
        self.controller = controller
        self.maxpool = nn.MaxPool2d(kernel_size = kernel_size, stride = stride)

    def forward(self, x):
        xmax = self.maxpool(x)
        out = torch.sum(torch.exp(self.controller * (x - xmax)), dim = (-2, -1), keepdim=True) / torch.prod(torch.tensor(x[0].shape))
        out  = xmax + torch.log(out) / self.controller
        return out