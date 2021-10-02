import torch
import cv2
import math
import numpy as np
from torchvision.transforms.functional import normalize as normalize_tv

class CustomTransform:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__

    def __eq__(self, name):
        return str(self) == name

    def __iter__(self):
        def iter_fn():
            for t in [self]:
                yield t
        return iter_fn()

    def __contains__(self, name):
        for t in self.__iter__():
            if isinstance(t, Compose):
                if name in t:
                    return True
            elif name == t:
                return True
        return False

class Compose(CustomTransform):
    """
    All transform in Compose should be able to accept two non None variable, img and boxes
    """
    def __init__(self, *transforms):
        self.transforms = [*transforms]

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __iter__(self):
        return iter(self.transforms)

    def modules(self):
        yield self
        for t in self.transforms:
            if isinstance(t, Compose):
                for _t in t.modules():
                    yield _t
            else:
                yield t

class Resize(CustomTransform):
    def __init__(self, size):
        if isinstance(size, int):
            size = (size, size)
        self.size = size

    def __call__(self, img):
        return cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)

    def reset_size(self, size):
        if isinstance(size, int):
            size = (size, size)
        self.size = size

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

    def __str__(self):
        return self.__class__.__name__ + '{}'.format(self.size[0] if isinstance(self.size, tuple) else self.size)

class CenterCrop(torch.nn.Module):
    def __init__(self, size):
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        
    @staticmethod
    def get_params(img, size):
        height, width = img.shape[0], img.shape[1]
        
        crop_width = size[0] if size[0] < width else width
        crop_height = size[1] if size[1] < height else height
        
        center_x, center_y = int(width/2), int(height/2)
        crop_width, crop_height = int(crop_width/2), int(crop_height/2) 
        return center_x, center_y, crop_width, crop_height

    def __call__(self, img):
        center_x, center_y, crop_width, crop_height = self.get_params(img, self.size)
        return img[center_y-crop_height:center_y+crop_height, center_x-crop_width:center_x+crop_width]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

    def __str__(self):
        return self.__class__.__name__ + '{}'.format(self.size[0] if isinstance(self.size, tuple) else self.size)


class RandomHorizontalFlip(CustomTransform):
    def __init__(self, prob=0.5):
        """
        Arguments:
        ----------
        prob_x: range [0, 1], probability to use horizontal flip
        """
        self.prob = prob

    def __call__(self, img):
        if np.random.random(1) < self.prob:
            img = np.ascontiguousarray(np.flip(img, axis=1))
        return img

class RandomResizedCrop(CustomTransform):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=cv2.INTER_LINEAR):
        
        if isinstance(size, int):
            size = (size, size)
        self.size = size

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
    
    @staticmethod
    def get_params(img, scale, ratio):
        height, width = img.shape[0], img.shape[1]
        area = height * width

        log_ratio = np.log(ratio)
        for _ in range(10):
            target_area = area * np.random.uniform(scale[0], scale[1])
            aspect_ratio = np.exp(np.random.uniform(log_ratio[0], log_ratio[1]))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = np.random.randint(0, height - h + 1)
                j = np.random.randint(0, width - w + 1)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w
        
    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return cv2.resize(img[i:i+h, j:j+w], self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string

class ToTensor(CustomTransform):
    def __init__(self, dtype=torch.float, grayscale=True):
        self.dtype=dtype
        self.grayscale = grayscale

    def __call__(self, img):
        if self.grayscale:
            img = np.expand_dims(img, axis=0)
        else:
            img = img.transpose(2, 0, 1)

        return torch.from_numpy(img).type(self.dtype) / 255.

class Normalize(CustomTransform):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        return normalize_tv(tensor, self.mean, self.std, self.inplace)

class DynamicNormalize(CustomTransform):
    def __init__(self, inplace=False):
        self.inplace = inplace

    def __call__(self, tensor):
        return normalize_tv(tensor, tensor.mean(), tensor.std().clamp(min=1e-7), inplace = self.inplace)

class UnNormalize(CustomTransform):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        out_tensor = torch.empty(0, dtype = tensor.dtype, device = tensor.device)
        for t, m, s in zip(tensor, self.mean, self.std):
            out_tensor = torch.cat((out_tensor, t.mul(s).add(m).unsqueeze(0)), 0)
        return out_tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)