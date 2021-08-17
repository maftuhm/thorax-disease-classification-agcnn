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

def main():
    pin_memory = True
    print(' pin_memory is', pin_memory)
    train_data = ChestXrayDataSet(data_dir = DATA_DIR, split = 'train', num_classes = 14, transform = transform_train)

    for num_workers in range(0, 20, 1): 
        train_loader = DataLoader(train_data, batch_size=16, num_workers=num_workers, pin_memory=pin_memory)
        start = time.time()
        for epoch in range(1, 5):
            progressbar = tqdm(range(len(train_loader)))
            for i, data in enumerate(train_loader):
                progressbar.update(1)
            progressbar.close()
        del train_loader
        end = time.time()
        print(" Finish with: {} seconds, num_workers={}\n".format(end - start, num_workers))

if __name__ == "__main__":
    main()