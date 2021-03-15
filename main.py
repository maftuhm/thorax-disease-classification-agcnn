import os
import os.path as path
import json
import argparse
from tqdm import tqdm
import shutil
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

from read_data import ChestXrayDataSet
from model import Net, FusionNet

cudnn.benchmark = True

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--exp_dir", type=str, default="./experiments/exp0")
	parser.add_argument("--resume", "-r", action="store_true")
	args = parser.parse_args()
	return args
args = parse_args()

exp_dir = args.exp_dir
exp_name = exp_dir.split('/')[-1]

with open(path.join(exp_dir, "cfg.json")) as f:
	exp_cfg = json.load(f)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

transform = transforms.Compose([
   transforms.Resize(tuple(exp_cfg['dataset']['resize'])),
   transforms.RandomResizedCrop(tuple(exp_cfg['dataset']['crop'])),
   transforms.ToTensor(),
   normalize,
])

DATASET_PATH = path.join('D://', 'Data', 'data')
IMAGES_DATA_PATH = path.join(DATASET_PATH, 'images')
TRAIN_IMAGE_LIST = path.join(DATASET_PATH, 'labels', 'dummy_data_train_list.txt')
VAL_IMAGE_LIST = path.join(DATASET_PATH, 'labels', 'dummy_data_val_list.txt')

MAX_BATCH_CAPACITY = 2
# batch_multiplier = exp_cfg['batch_size']['global'] // MAX_BATCH_CAPACITY

train_dataset = ChestXrayDataSet(data_dir = IMAGES_DATA_PATH, image_list_file = TRAIN_IMAGE_LIST, transform = transform)
train_loader = DataLoader(dataset = train_dataset, batch_size = MAX_BATCH_CAPACITY, shuffle = True, num_workers = 4)

val_dataset = ChestXrayDataSet(data_dir = IMAGES_DATA_PATH, image_list_file = VAL_IMAGE_LIST, transform = transform)
val_loader = DataLoader(dataset = val_dataset, batch_size = MAX_BATCH_CAPACITY, shuffle = False, num_workers = 4)

BRANCH_NAME_LIST = ['global', 'local', 'fusion']
BEST_VAL_LOSS = {
	'global': 1000,
	'local': 1000,
	'fusion': 1000
}

#-------------------- SETTINGS: MODEL
GlobalModel = Net(exp_cfg['backbone'])
LocalModel = Net(exp_cfg['backbone'])
FusionModel = FusionNet(exp_cfg['backbone'])

#-------------------- SETTINGS: OPTIMIZER & SCHEDULER
optimizer_global = optim.SGD(GlobalModel.parameters(), **exp_cfg['optimizer']['SGD'])
optimizer_local = optim.SGD(LocalModel.parameters(), **exp_cfg['optimizer']['SGD'])
optimizer_fusion = optim.SGD(FusionModel.parameters(), **exp_cfg['optimizer']['SGD'])

lr_scheduler_global = optim.lr_scheduler.StepLR(optimizer_global , **exp_cfg['lr_scheduler'])
lr_scheduler_local = optim.lr_scheduler.StepLR(optimizer_local , **exp_cfg['lr_scheduler'])
lr_scheduler_fusion = optim.lr_scheduler.StepLR(optimizer_fusion , **exp_cfg['lr_scheduler'])

#-------------------- SETTINGS: LOSS FUNCTION
loss_global = nn.BCELoss()
loss_local = nn.BCELoss()
loss_fusion = nn.BCELoss()

def heatmap_crop_origin(heatmap, origin_img, threshold = 0.7):

	batchsize = heatmap.size(0)
	img = torch.randn(batchsize, 3, 224, 224)

	for batch in range(batchsize):
		heatmap_one = torch.abs(heatmap[batch])
		heatmap_two = torch.max(heatmap_one, dim=0)[0].squeeze(0)
		max1 = torch.max(heatmap_two)
		min1 = torch.min(heatmap_two)

		heatmap_two = (heatmap_two - min1) // (max1 - min1)
		heatmap_two[heatmap_two > threshold] = 1
		heatmap_two[heatmap_two != 1] = 0

		where = torch.from_numpy(np.argwhere(heatmap_two.detach().numpy() == 1))
		xmin = int((torch.min(where, dim =0)[0][0])*224//7)
		xmax = int(torch.max(where, dim=0)[0][0]*224//7)
		ymin = int(torch.min(where, dim =0)[0][1]*224//7)
		ymax = int(torch.max(where, dim =0)[0][1]*224//7)

		if xmin == xmax:
			xmin = int((torch.min(where, dim =0)[0][0])*224//7)
			xmax = int((torch.max(where, dim=0)[0][0] + 1)*224//7)
		if ymin == ymax:
			ymin = int((torch.min(where, dim =0)[0][1])*224//7)
			ymax = int((torch.max(where, dim =0)[0][1] + 1)*224//7)

		sliced = transforms.ToPILImage()(origin_img[batch][:, xmin:xmax, ymin:ymax])
		img_one = sliced.resize((224, 224), Image.ANTIALIAS)
		img[batch] = transforms.ToTensor()(img_one)

	return img

def save_model(epoch, val_loss, model, optimizer, lr_scheduler, branch_name = 'global'):
	save_dict = {
		"epoch": epoch,
		'loss_value' : val_loss,
		"net": model.state_dict(),
		"optim": optimizer.state_dict(),
		"lr_scheduler": lr_scheduler.state_dict()
	}
	save_name = path.join(exp_dir, exp_dir.split('/')[-1] + '_' + branch_name + '.pth')
	torch.save(save_dict, save_name)
	print(" Model is saved: {}".format(save_name))

def train(epoch, branch_name, model, data_loader, optimizer, lr_scheduler, loss_func, test_model, batch_multiplier):

	print(" Training {} model using lr = {}".format(branch_name, lr_scheduler.get_last_lr()))

	model.train()
	count = 0
	# count_batch = 0
	running_loss = 0

	progressbar = tqdm(range(len(data_loader)))

	for i, (image, target) in enumerate(data_loader):
		
		optimizer.zero_grad()
		# if count_batch == 0:
		# 	optimizer.step()
		# 	optimizer.zero_grad()
		# 	count_batch = batch_multiplier

		if branch_name == 'local':
			with torch.no_grad():
				output, heatmap, pool = test_model(image.cuda())
				image = heatmap_crop_origin(heatmap.cpu(), image, exp_cfg['threshold'])

			del output, heatmap, pool
			torch.cuda.empty_cache()

		if branch_name == 'fusion':
			with torch.no_grad():
				output, heatmap, pool1 = test_model[0](image.cuda())
				inputs = heatmap_crop_origin(heatmap.cpu(), image)
				output, heatmap, pool2 = test_model[1](inputs.cuda())
				pool1 = pool1.view(pool1.size(0), -1)
				pool2 = pool2.view(pool2.size(0), -1)
				image = torch.cat((pool1.cpu(), pool2.cpu()), dim=1)

			del output, heatmap, pool1, pool2, inputs
			torch.cuda.empty_cache()

		image = image.cuda()
		target = target.cuda()

		output, heatmap, pool = model(image)
		loss = loss_func(output, target) / batch_multiplier

		loss.backward()
		optimizer.step()

		running_loss += loss.data.item() * batch_multiplier
		count += 1
		# count_batch -= 1

		progressbar.set_description(" Epoch: [{}/{}] | loss: {:.5f}".format(epoch, exp_cfg['NUM_EPOCH'] - 1, loss.data.item() * batch_multiplier))
		progressbar.update(1)
	progressbar.close()

	lr_scheduler.step()

	epoch_train_loss = float(running_loss) / float(count)
	print(' Epoch over Loss: {:.5f}'.format(epoch_train_loss))

	save_model(epoch, epoch_train_loss, model, optimizer, lr_scheduler, branch_name)
	print()

	del image, target, loss, epoch_train_loss, output, heatmap, pool, model
	torch.cuda.empty_cache()



def val(epoch, branch_name, model, data_loader, optimizer, loss_func, test_model):

	print(" Validating {} model".format(branch_name, epoch,))

	model.eval()
	
	count = 0
	running_loss = 0
	
	progressbar = tqdm(range(len(data_loader)))

	with torch.no_grad():
		for i, (image, target) in enumerate(data_loader):

			if branch_name == 'local':
				output, heatmap, pool = test_model(image.cuda())
				image = heatmap_crop_origin(heatmap.cpu(), image, exp_cfg['threshold'])

				del output, heatmap, pool

			if branch_name == 'fusion':
				output, heatmap, pool1 = test_model[0](image.cuda())
				inputs = heatmap_crop_origin(heatmap.cpu(), image)
				output, heatmap, pool2 = test_model[1](inputs.cuda())
				pool1 = pool1.view(pool1.size(0), -1)
				pool2 = pool2.view(pool2.size(0), -1)
				image = torch.cat((pool1.cpu(), pool2.cpu()), dim=1)

				del output, heatmap, pool1, pool2, inputs

			image = image.cuda()
			target = target.cuda()

			output, heatmap, pool = model(image)

			loss = loss_func(output, target)

			running_loss += loss.data.item()
			count += 1

			progressbar.set_description(" Epoch: [{}/{}] | loss: {:.5f}".format(epoch,  exp_cfg['NUM_EPOCH'] - 1, loss.data.item()))
			progressbar.update(1)
		progressbar.close()
		
		epoch_val_loss = float(running_loss) / float(count)
		print(' Epoch over Loss: {:.5f}'.format(epoch_val_loss))

		if epoch_val_loss < BEST_VAL_LOSS[branch_name]:
			BEST_VAL_LOSS[branch_name] = epoch_val_loss
			save_name = path.join(exp_dir, exp_dir.split('/')[-1] + '_' + branch_name + '.pth')
			copy_name = os.path.join(exp_dir, exp_dir.split('/')[-1] + '_' + branch_name + '_best.pth')
			shutil.copyfile(save_name, copy_name)
			print(" Best model is saved: {}\n".format(copy_name))

	return epoch_val_loss

def main():
	global GlobalModel, LocalModel, FusionModel

	for branch_name in BRANCH_NAME_LIST:

		batch_multiplier = exp_cfg['batch_size'][branch_name] // MAX_BATCH_CAPACITY

		train_dataset = ChestXrayDataSet(data_dir = IMAGES_DATA_PATH, image_list_file = TRAIN_IMAGE_LIST, transform = transform)
		train_loader = DataLoader(dataset = train_dataset, batch_size = MAX_BATCH_CAPACITY, shuffle = True, num_workers = 4)

		val_dataset = ChestXrayDataSet(data_dir = IMAGES_DATA_PATH, image_list_file = VAL_IMAGE_LIST, transform = transform)
		val_loader = DataLoader(dataset = val_dataset, batch_size = exp_cfg['batch_size'][branch_name], shuffle = False, num_workers = 4)

		if args.resume:
			save_dict = torch.load(os.path.join(exp_dir, exp_dir.split('/')[-1] + '_'+ branch_name + '.pth'))

			if branch_name == 'global':
				GlobalModel = GlobalModel.to(device)
				GlobalModel.load_state_dict(save_dict['net'])
				optimizer_global.load_state_dict(save_dict['optim'])
				lr_scheduler_global.load_state_dict(save_dict['lr_scheduler'])

			if branch_name == 'local':
				LocalModel = LocalModel.to(device)
				LocalModel.load_state_dict(save_dict['net'])
				optimizer_local.load_state_dict(save_dict['optim'])
				lr_scheduler_local.load_state_dict(save_dict['lr_scheduler'])

			if branch_name == 'fusion':
				FusionModel = FusionModel.to(device)
				FusionModel.load_state_dict(save_dict['net'])
				optimizer_fusion.load_state_dict(save_dict['optim'])
				lr_scheduler_fusion.load_state_dict(save_dict['lr_scheduler'])

			BEST_VAL_LOSS[branch_name] = save_dict['loss_value']
			start_epoch = save_dict['epoch'] + 1
		else:
			start_epoch = 0

		for epoch in range(start_epoch, exp_cfg['NUM_EPOCH']):

			if branch_name == 'global':
				GlobalModel = GlobalModel.to(device)
				train(epoch, branch_name, GlobalModel, train_loader, optimizer_global, lr_scheduler_global, loss_global, None, batch_multiplier)
				val(epoch, branch_name, GlobalModel, val_loader, optimizer_global, loss_global, None)

			if branch_name == 'local':

				save_dict_global = torch.load(os.path.join(exp_dir, exp_dir.split('/')[-1] + '_global_best' + '.pth'))
	
				GlobalModelTest = Net(exp_cfg['backbone']).to(device)
				GlobalModelTest.load_state_dict(save_dict_global['net'])

				LocalModel = LocalModel.to(device)
				train(epoch, branch_name, LocalModel, train_loader, optimizer_local, lr_scheduler_local, loss_local, GlobalModelTest, batch_multiplier)
				val(epoch, branch_name, LocalModel, val_loader, optimizer_local, loss_local, GlobalModelTest)
			
			if branch_name == 'fusion':

				save_dict_global = torch.load(os.path.join(exp_dir, exp_dir.split('/')[-1] + '_global_best' + '.pth'))
				save_dict_local = torch.load(os.path.join(exp_dir, exp_dir.split('/')[-1] + '_local_best' + '.pth'))

				GlobalModelTest = Net(exp_cfg['backbone']).to(device)
				LocalModelTest = Net(exp_cfg['backbone']).to(device)

				GlobalModelTest.load_state_dict(save_dict_global['net'])
				LocalModelTest.load_state_dict(save_dict_local['net'])

				FusionModel = FusionModel.to(device)
				train(epoch, branch_name, FusionModel, train_loader, optimizer_fusion, lr_scheduler_fusion, loss_fusion, (GlobalModelTest, LocalModelTest), batch_multiplier)
				val(epoch, branch_name, FusionModel, val_loader, optimizer_fusion, loss_fusion, (GlobalModelTest, LocalModelTest))

if __name__ == "__main__":
	main()
