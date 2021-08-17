import time
from tqdm import tqdm
from utils import ChestXrayDataSet
from torch.utils.data import DataLoader
from config import *
import torchvision.transforms as transforms

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

transform_train = transforms.Compose([
   transforms.Resize((256, 256)),
   transforms.RandomResizedCrop((224, 224)),
   transforms.RandomHorizontalFlip(),
   transforms.ToTensor(),
   normalize,
])


transform_test = transforms.Compose([
   transforms.Resize((256, 256)),
   transforms.CenterCrop((224, 224)),
   transforms.ToTensor(),
   normalize,
])

def loop(loader):
    for epoch in range(1, 5):
        progressbar = tqdm(range(len(loader)))
        for i, data in enumerate(loader):
            progressbar.update(1)
        progressbar.close()
    del loader

def main():
    train_data = ChestXrayDataSet(data_dir = DATA_DIR, split = 'test', num_classes = 14, transform = transform_train)
    val_data = ChestXrayDataSet(data_dir = DATA_DIR, split = 'val', num_classes = 14, transform = transform_test)
    for num_workers in range(3, 10): 
        train_loader = DataLoader(train_data, batch_size=16, shuffle = True, num_workers = num_workers, pin_memory = True)
        val_loader = DataLoader(val_data, batch_size = 64, shuffle = False, num_workers = num_workers, pin_memory = True)
        
        start = time.time()
        loop(train_loader)
        end = time.time()
        print(" Finish train_loader with: {} seconds, num_workers={}\n".format(end - start, num_workers))
        
        start = time.time()
        loop(val_loader)
        end = time.time()
        print(" Finish val_loader with: {} seconds, num_workers={}\n".format(end - start, num_workers))

if __name__ == "__main__":
    main()