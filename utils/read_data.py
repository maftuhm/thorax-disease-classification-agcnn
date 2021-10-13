import os
import json
import cv2
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

class ChestXrayDataSet(Dataset):
    
    def __init__(self, data_dir, image_set, num_classes=14, transform=None, init_transform = None, get_bbox = False):

        assert image_set in {'train', 'val', 'test'}, "image_set is not valid!"
        self.data_dir_path = data_dir
        self.image_set = image_set
        self.num_classes = num_classes
        self.transform = transform
        self.init_transform = init_transform
        self.return_bbox = get_bbox

        self.static_images_dir = 'images_' + str(init_transform).lower()

        if not os.path.exists(os.path.join(data_dir, self.static_images_dir)):
            print("Static Image is going to get generated into dir: {} ...".format(self.static_images_dir))
            self.create_static_images()
            print("Static Image is successfully generated into dir: {} ...".format(self.static_images_dir))

        self.create_index()

    def create_static_images(self):

        os.makedirs(os.path.join(self.data_dir_path, self.static_images_dir))
        
        images_dir_path = os.path.join(self.data_dir_path, 'images')
        images_list = os.listdir(images_dir_path)

        progressbar = tqdm(range(len(images_list)))
        for name in images_list:
            image = cv2.imread(os.path.join(images_dir_path, name), cv2.IMREAD_GRAYSCALE)

            if self.init_transform is not None:
                image = self.init_transform(image)

            progressbar.set_description("Generating {} image".format(name))
            cv2.imwrite(os.path.join(self.data_dir_path, self.static_images_dir, name), image)
            progressbar.update(1)
        progressbar.close()
    
    def create_index(self):
        self.images = []
        self.labels = []
        self.bboxes = []

        # prefix = 'my_'
        # suffix = '_image-patient-wise'            
        prefix = ''
        suffix = ''
        if self.return_bbox:
            suffix += '_with_bbox' 

        json_raw_files = os.path.join(self.data_dir_path, 'labels', prefix + self.image_set + '_list' + suffix + '.json')
        with open(json_raw_files, "r") as file:
            for data in file:
                data = json.loads(data)
                self.images.append(os.path.join(self.data_dir_path, self.static_images_dir, data['index']))
                self.labels.append(data['label'][:self.num_classes])
                self.bboxes.append(data.get('bbox', []))

    def __getitem__(self, index):
        image = cv2.imread(self.images[index], cv2.IMREAD_GRAYSCALE)
        label = self.labels[index]
        bbox = self.bboxes[index]

        if self.transform is not None:
            image = self.transform(image)

        if self.return_bbox:
            return image, torch.FloatTensor(label), bbox

        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.labels)