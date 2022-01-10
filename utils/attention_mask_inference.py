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
    def __init__(self, weights = None, threshold = 0, distance_function = None, size = 224, keepratio = True):

        self.weights = weights

        L1 = lambda features: np.abs(features).sum(1)
        L2 = lambda features: np.sqrt((features ** 2).sum(1))
        Lmax = lambda features: np.abs(features).max(1)

        if distance_function == "Lmax":
            self.distance = lambda images: self._normalize(Lmax(images))
        elif distance_function == "L1":
            self.distance = lambda images: self._normalize(L1(images))
        elif distance_function == "L2":
            self.distance = lambda images: self._normalize(L2(images))
        else:
            raise Exception("distance function must be L1, L2 or Lmax")

        self.resize = lambda images: np.asarray([cv2.resize(img, (size, size)) for img in images])
        self.threshold = threshold
        self.binary_images = lambda images: np.asarray([self.max_connected(self._binary(img, threshold)) for img in images])
        self.touint8 = lambda img: img if img.dtype == np.uint8 else np.uint8(img * 255)
        self.get_coords = lambda images: [self.get_nonzero_coords(img) for img in images]
        self.crop_resize = lambda images, bboxes: np.asarray([self._resize(self._crop(img, bbox), size, keepratio = keepratio) for img, bbox in zip(images, bboxes)])

    def __call__(self, x, features):
        features = np.asarray(features) if self.weights is None else self._add_weight(features, self.weights) # N x D x H x W
        heatmaps = self.distance(features)   # N x H x W
        out_heatmaps = self.resize(heatmaps)  # N x 224 x 224
        heatmaps = self.touint8(out_heatmaps)
        heatmaps_bin = self.binary_images(heatmaps) 
        coords = self.get_coords(heatmaps_bin)
        out = self.crop_resize(x.numpy().squeeze(), coords)
        out = torch.from_numpy(out).unsqueeze(1)
        result = {
            'image' : out,
            'heatmap' : out_heatmaps,
            'coordinate' : coords
        }
        return result

    @staticmethod
    def _resize(image, size, inter = cv2.INTER_LINEAR, background = -2.1760, keepratio = True):

        if not keepratio:
            dim = (size, size)
            return cv2.resize(image, dim, interpolation = inter)

        dim = None
        (h, w) = image.shape[:2]
        if h > w:
            r = size / float(h)
            dim = (round(w * r), size)
        else:
            r = size / float(w)
            dim = (size, round(h * r))

        resized = cv2.resize(image, dim, interpolation = inter)

        (h, w) = resized.shape[:2]
        dim = (size, size) if resized.ndim == 2 else (size, size, resized.shape[-1])
        new_img = np.ones(dim) * background
        new_img = new_img.astype(resized.dtype)

        if size > h:
            dif = round((size - h) / 2)
            new_img[dif:h+dif,:] = resized
        else:
            dif = round((size - w) / 2)
            new_img[:,dif:w+dif] = resized

        return new_img

    @staticmethod
    def _add_weight(features, weight):
        weight = weight if type(weight) == np.ndarray else np.asarray(weight)
        features = features if type(features) == np.ndarray else np.asarray(features)
        features = np.transpose(features, (0, 2, 3, 1))
        weight = np.transpose(weight, (1, 0))

        assert weight.shape[0] == features.shape[-1], "shape features and weight are not match"

        return np.transpose(np.matmul(features, weight), (0, 3, 1, 2))

    @staticmethod
    def _normalize(images):
        min_ = images.min(axis = (-2, -1), keepdims=True)
        max_ = images.max(axis = (-2, -1), keepdims=True)
        return (images - min_) / (max_ - min_)
    
    @staticmethod
    def _binary(heatmap, threshold):
        assert heatmap.dtype == np.uint8, "heatmap dtype must be np.uint8"

        if threshold != 0:
            if isinstance(threshold, float):
                _range = (round(threshold[0] * 255), 255)
            elif isinstance(threshold, int):
                _range = (threshold, 255)
            else:
                _range = threshold if isinstance(threshold[0], int) else (round(threshold[0] * 255), round(threshold[1] * 255))

            return cv2.threshold(heatmap, _range[0], _range[1], cv2.THRESH_BINARY)[1]

        return cv2.threshold(heatmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    @staticmethod
    def get_nonzero_coords(heatmap):
        ind = np.transpose(heatmap.nonzero())
        return ind[:,1].min(), ind[:,0].min(), ind[:,1].max(), ind[:,0].max()

    @staticmethod
    def _crop(image, bbox):
        return image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    @staticmethod
    def max_connected(heatmap):
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

    def load_weight(self, weight):
        self.weights = weight if type(weight) == np.ndarray else np.asarray(weight)
        print(" Fully connected global weight is loaded.")
        return self.weights != None

class _AttentionMaskInferenceOld_:
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
            xmin = torch.min(coord[:,0]).item()
            xmax = torch.max(coord[:,0]).item()
            ymin = torch.min(coord[:,1]).item()
            ymax = torch.max(coord[:,1]).item()
            img = images[i][:, xmin:xmax, ymin:ymax]
            out[i] = self.resize(img.unsqueeze(0)).squeeze(0)

            coords.append([ymin, xmin, ymax, xmax])

        return out, coords