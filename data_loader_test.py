import time
from tqdm import tqdm
from utils import ChestXrayDataSet
from torch.utils.data import DataLoader
from config import *
from utils import transforms

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

def loop(loader, non_blocking = False):
    for epoch in range(0, 3):
        progressbar = tqdm(range(len(loader)))
        for i, (images, labels) in enumerate(loader):
            images = images.to('cuda:0', non_blocking=non_blocking)
            labels = labels.to('cuda:0', non_blocking=non_blocking)
            progressbar.update(1)
        progressbar.close()
    del loader

def main():

    # normalize = transforms.Normalize(
    #    mean=[0.485, 0.456, 0.406],
    #    std=[0.229, 0.224, 0.225]
    # )

    transform_init = transforms.Resize((256, 256))

    transform_train = transforms.Compose(
       transforms.RandomResizedCrop((224, 224)),
       transforms.RandomHorizontalFlip(),
       transforms.ToTensor(),
       transforms.DynamicNormalize()
    )


    transform_test = transforms.Compose(
       transforms.CenterCrop((224, 224)),
       transforms.ToTensor(),
       transforms.DynamicNormalize()
    )
    # model = ResAttCheXNet(pretrained = True, num_classes = 14).to('cuda:0')
    # criterion = nn.BCELoss()

    best_time_loader = 1000.
    best_num_worker = -1

    train_data = ChestXrayDataSet(DATA_DIR, 'train', num_classes = 14, transform = transform_train, init_transform=transform_init)
    # val_data = ChestXrayDataSet(data_dir = DATA_DIR, split = 'val', num_classes = 14, transform = transform_test, init_transform=transform_init)
    pin_memory = True
    non_blocking = True
    print(" Start loader with pin_memory={} and non_blocking={}".format(pin_memory, non_blocking))
    for num_workers in range(4, 10): 
        train_loader = DataLoader(train_data, batch_size=20, shuffle = True, num_workers = num_workers, pin_memory = pin_memory)
        # val_loader = DataLoader(val_data, batch_size = 64, shuffle = False, num_workers = num_workers, pin_memory = pin_memory)
        
        start = time.time()
        loop(train_loader, non_blocking)
        end = time.time()
        delta = end - start
        print(" Finish train_loader with: {} seconds, num_workers={}\n".format(delta, num_workers))
        if delta < best_time_loader:
            best_time_loader = delta
            best_num_worker = num_workers


        # start = time.time()
        # loop(val_loader)
        # end = time.time()
    print(" Finish train_loader with best result: {} seconds, num_workers={}\n".format(best_time_loader, best_num_worker))

if __name__ == "__main__":
    main()