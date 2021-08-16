import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, split, num_classes=14, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        assert split in {'train', 'test', 'val'}
        
        image_names = []
        labels = []
        with open(os.path.join(data_dir, 'labels', 'my_' + split + '_list.txt'), "r") as f:
            for line in f:
                items = line.split()
                image_name= items[0]
                label = items[1:num_classes+1]
                label = [int(i) for i in label]
                image_name = os.path.join(data_dir, 'images', image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)

