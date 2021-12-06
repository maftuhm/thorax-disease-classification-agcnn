import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2
from skimage.measure import label

def L1(feature, axis = 0):
    output = torch.abs(feature)
    output = torch.sum(output, axis = axis) / feature.shape[axis]
    return output

def L2(feature, axis = 0):
    output = torch.sum(feature ** 2, axis = axis)
    output = torch.sqrt(output) / feature.shape[axis]
    return output

def Lmax(feature, axis = 0):
    output = torch.abs(feature)
    output = torch.max(output, dim = axis)[0]
    return output

def normalize01(x):
    n, h, w = x.shape
    out = x.view(n, -1)
    min1 = out.min(1, keepdim=True)[0]
    max1 = out.max(1, keepdim=True)[0]
    out = (out - min1) / (max1 - min1)
    out = out.view(n, h, w)
    return out

def selectMaxConnect(heatmap):
    labeled_img, num = label(heatmap, connectivity=2, background=0, return_num=True)
    max_label = 0
    max_num = 0
    for i in range(1, num+1):
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label)
    if max_num == 0:
        lcc = (labeled_img == -1)
    lcc = lcc + 0
    return lcc 


def binImage(heatmap, threshold = 0.7, thresh_otsu = False):
    if thresh_otsu:
        _, heatmap_bin = cv2.threshold(heatmap , 0 , 255 , cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        _, heatmap_bin = cv2.threshold(heatmap , int(255 * threshold) , 255 , cv2.THRESH_BINARY)
    return heatmap_bin

class AttentionMaskInference:
    def __init__(self, threshold = 0.7, distance_function = "Lmax", size = (224, 224), mode = 'bilinear'):

        if distance_function == "Lmax":
            self.distance = lambda x: normalize01(Lmax(x, 1))
        elif distance_function == "L1":
            self.distance = lambda x: normalize01(L1(x, 1))
        elif distance_function == "L2":
            self.distance = lambda x: normalize01(L2(x, 1))
        else:
            raise Exception("distance function must be L1, L2 or Lmax")

        self.resize = nn.Upsample(size=size, mode = mode, align_corners = True)
        self.threshold = threshold

    def __call__(self, x, features):
        heatmap = self.distance(features)
        out_heatmap = self.resize(heatmap.unsqueeze(1)).squeeze(1)

        if isinstance(self.threshold, tuple) or isinstance(self.threshold, list):
            # still error
            heatmap[heatmap < self.threshold[0]] = 0.
            heatmap[heatmap > self.threshold[1]] = 0.
            heatmap[heatmap != 0.] = 1.
        else:
            heatmap = (out_heatmap > self.threshold).float()

        out, coords = self.crop_resize(x, heatmap)

        result = {
            'image' : out,
            'heatmap' : out_heatmap,
            'coordinate' : coords
        }
        return result
    
    def crop_resize(self, images, heatmaps):

        out = torch.zeros_like(images)
        coords = []
        for i in range(images.size(0)):
            heatmap = heatmaps[i] * torch.from_numpy(selectMaxConnect(heatmaps[i]))
            coord = torch.nonzero(heatmap, as_tuple = False)
            xmin = torch.min(coord[:,0])
            xmax = torch.max(coord[:,0])
            ymin = torch.min(coord[:,1])
            ymax = torch.max(coord[:,1])
            img = images[i][:, xmin:xmax, ymin:ymax]
            out[i] = self.resize(img.unsqueeze(0)).squeeze(0)

            coords.append([ymin, xmin, ymax, xmax])

        return out, coords