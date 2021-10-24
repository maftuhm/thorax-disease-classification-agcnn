import time
import argparse
from tqdm import tqdm
from utils import ChestXrayDataSet
from torch.utils.data import DataLoader
from config import *
from utils import transforms

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser(description='AG-CNN')
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--pin_memory", "-pm", action="store_true")
    parser.add_argument("--non_blocking", "-nb", action="store_true")
    parser.add_argument("--start_nw", type=int, default=2)
    parser.add_argument("--end_nw", type=int, default=8)
    parser.add_argument("--data_split", type=str, default='val')
    args = parser.parse_args()
    return args

args = parse_args()

def loop(loader, non_blocking = False):
    for epoch in range(0, 3):
        progressbar = tqdm(range(len(loader)))
        for i, (images, labels) in enumerate(loader):
            images = images.to('cuda:0', non_blocking=non_blocking)
            labels = labels.to('cuda:0', non_blocking=non_blocking)
            progressbar.update(1)
        progressbar.close()
    del loader, images, labels
    torch.cuda.empty_cache()

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

    train_data = ChestXrayDataSet(DATA_DIR, args.data_split, num_classes = 15, transform = transform_train, init_transform=transform_init)
    # val_data = ChestXrayDataSet(data_dir = DATA_DIR, split = 'val', num_classes = 14, transform = transform_test, init_transform=transform_init)

    print(" Start loader with pin_memory={} and non_blocking={}".format(args.pin_memory, args.non_blocking))
    for num_workers in range(args.start_nw, args.end_nw + 1): 
        train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = False, num_workers = num_workers, pin_memory = args.pin_memory)
        # val_loader = DataLoader(val_data, batch_size = 64, shuffle = False, num_workers = num_workers, pin_memory = pin_memory)
        
        start = time.time()
        loop(train_loader, args.non_blocking)
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