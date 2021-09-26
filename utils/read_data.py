import os
import cv2
import torch
import numpy as np
import sharearray
from tqdm import tqdm
from torch.utils.data import Dataset
from numpy.lib.format import open_memmap

class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, split, num_classes=14, transform=None, init_transform = None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        assert split in {'train', 'test', 'val', 'train_val'}

        array_dir = os.path.join(data_dir, 'npy')
        os.makedirs(array_dir, exist_ok=True)

        filename        = 'my_array_' + split + '_images'
        filename_list   = 'my_' + split + '_list'

        array_file      = os.path.join(array_dir, filename + '.npy')
        data_list       = os.path.join(data_dir, 'labels', filename_list + '.txt')

        if os.path.isfile(array_file) is not True:
            images = sharearray.cache(
                filename,
                lambda: self.create_data_array(data_dir, data_list, init_transform),
                shm_path=array_dir,
                prefix=''
            )
            images.flush()
            del images

        self.images = open_memmap(array_file, mode='r+')

        labels = []
        with open(data_list, "r") as file:
            lines = file.readlines()
            for line in lines:
                items = line.split()
                label = items[1:num_classes+1]
                label = [int(i) for i in label]
                labels.append(label)

        self.labels = np.asarray(labels)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        image = np.ascontiguousarray(np.asarray(self.images[index], dtype=np.float32))
        # image = np.expand_dims(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), axis=2)
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.labels)

    def create_data_array(self, data_dir, data_list, transform):

        images = []

        with open(data_list, "r") as file:

            lines = file.readlines()
            progressbar = tqdm(range(len(lines)))
            for line in lines:
                items = line.split()
                image_name= items[0]
                image_dir = os.path.join(data_dir, 'images', image_name)
                image = cv2.imread(image_dir)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                if transform is not None:
                    image = transform(image)

                images.append(image)
                progressbar.update(1)
            progressbar.close()

        return np.asarray(images)

