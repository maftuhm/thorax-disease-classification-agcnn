import os
import os.path as path
import json
import argparse
from tqdm import tqdm
import shutil
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
from skimage.measure import label

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

from read_data import ChestXrayDataSet
from model import Net, FusionNet

cudnn.benchmark = True

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--exp_dir", type=str, default="./experiments/exp4")
	parser.add_argument("--resume", "-r", action="store_true")
	args = parser.parse_args()
	return args
args = parse_args()

exp_dir = args.exp_dir
exp_name = exp_dir.split('/')[-1]

with open(path.join(exp_dir, "cfg.json")) as f:
	exp_cfg = json.load(f)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(exp_dir + '/log')

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


CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
				'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

DATASET_PATH = path.join('..', 'lung-disease-detection', 'data')
IMAGES_DATA_PATH = path.join(DATASET_PATH, 'images')
TRAIN_IMAGE_LIST = path.join(DATASET_PATH, 'labels', 'train_list.txt')
VAL_IMAGE_LIST = path.join(DATASET_PATH, 'labels', 'val_list.txt')

MAX_BATCH_CAPACITY = {
	'global' : 16,
	'local' : 8,
	'fusion' : 8
}
# batch_multiplier = exp_cfg['batch_size']['global'] // MAX_BATCH_CAPACITY

# train_dataset = ChestXrayDataSet(data_dir = IMAGES_DATA_PATH, image_list_file = TRAIN_IMAGE_LIST, transform = transform)
# train_loader = DataLoader(dataset = train_dataset, batch_size = MAX_BATCH_CAPACITY, shuffle = True, num_workers = 4)

# val_dataset = ChestXrayDataSet(data_dir = IMAGES_DATA_PATH, image_list_file = VAL_IMAGE_LIST, transform = transform)
# val_loader = DataLoader(dataset = val_dataset, batch_size = exp_cfg['batch_size']['global'], shuffle = False, num_workers = 4)

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

		heatmap_two = (heatmap_two - min1) / (max1 - min1)
		heatmap_two[heatmap_two > threshold] = 1
		heatmap_two[heatmap_two != 1] = 0

		heatmap_maxconn = select_max_connect(heatmap_two.detach().numpy())

		where = torch.from_numpy(np.argwhere(heatmap_maxconn == 1))
		xmin = int((torch.min(where, dim =0)[0][0]) * 224 // 7)
		xmax = int(torch.max(where, dim=0)[0][0] * 224 // 7)
		ymin = int(torch.min(where, dim =0)[0][1] * 224 // 7)
		ymax = int(torch.max(where, dim =0)[0][1] * 224 // 7)

		if xmin == xmax:
			xmin = int((torch.min(where, dim =0)[0][0]) * 224 // 7)
			xmax = int((torch.max(where, dim=0)[0][0] + 1) * 224 // 7)
		if ymin == ymax:
			ymin = int((torch.min(where, dim =0)[0][1]) * 224 // 7)
			ymax = int((torch.max(where, dim =0)[0][1] + 1) * 224 // 7)

		sliced = transforms.ToPILImage()(origin_img[batch][:, xmin:xmax, ymin:ymax])
		img_one = sliced.resize((224, 224), Image.ANTIALIAS)
		img[batch] = transforms.ToTensor()(img_one)

	return img

def select_max_connect(heatmap):
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

def compute_AUCs(gt, pred):
	AUROCs = []
	gt_np = gt.cpu().numpy()
	pred_np = pred.cpu().numpy()
	for i in range(len(CLASS_NAMES)):
		AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
	return AUROCs

def train(epoch, branch_name, model, data_loader, optimizer, lr_scheduler, loss_func, test_model, batch_multiplier):

	print(" Training {} model using lr = {}".format(branch_name, lr_scheduler.get_last_lr()))

	model.train()
	count = 0
	running_loss = 0.
	mini_batch_loss = 0.

	progressbar = tqdm(range(len(data_loader)))

	for i, (image, target) in enumerate(data_loader):
		
		# optimizer.zero_grad()
		if (count + 1) % batch_multiplier == 0:
			optimizer.step()
			optimizer.zero_grad()

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
		# optimizer.step()

		running_loss += loss.data.item() * batch_multiplier
		mini_batch_loss += loss.data.item()
		count += 1

		progressbar.set_description(" Epoch: [{}/{}] | loss: {:.5f}".format(epoch, exp_cfg['NUM_EPOCH'] - 1, loss.data.item() * batch_multiplier))
		progressbar.update(1)

		if count % batch_multiplier == 0:
			writer.add_scalar('train/' + branch_name + ' loss', mini_batch_loss, epoch * len(data_loader) + i)
			mini_batch_loss = 0.

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
	
	gt = torch.FloatTensor().cuda()
	pred = torch.FloatTensor().cuda()

	count = 0
	running_loss = 0.
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
			gt = torch.cat((gt, target), 0)

			output, heatmap, pool = model(image)
			pred = torch.cat((pred, output.data), 0)

			loss = loss_func(output, target)

			running_loss += loss.data.item()
			count += 1

			progressbar.set_description(" Epoch: [{}/{}] | loss: {:.5f}".format(epoch,  exp_cfg['NUM_EPOCH'] - 1, loss.data.item()))
			progressbar.update(1)
			writer.add_scalar('val/' + branch_name + ' loss', loss.data.item(), epoch * len(data_loader) + i)

		progressbar.close()
		
		epoch_val_loss = float(running_loss) / float(count)
		print(' Epoch over Loss: {:.5f}'.format(epoch_val_loss))

		if epoch_val_loss < BEST_VAL_LOSS[branch_name]:
			BEST_VAL_LOSS[branch_name] = epoch_val_loss
			save_name = path.join(exp_dir, exp_dir.split('/')[-1] + '_' + branch_name + '.pth')
			copy_name = os.path.join(exp_dir, exp_dir.split('/')[-1] + '_' + branch_name + '_best.pth')
			shutil.copyfile(save_name, copy_name)
			print(" Best model is saved: {}".format(copy_name))

		AUROCs = compute_AUCs(gt, pred)
		print("|=======================================|")
		print("|\t\t  AUROC\t\t\t|")
		print("|=======================================|")
		print("|\t      global branch\t\t|")
		print("|---------------------------------------|")
		for i in range(len(CLASS_NAMES)):
			if len(CLASS_NAMES[i]) < 6:
				print("| {}\t\t\t|".format(CLASS_NAMES[i]), end="")
			elif len(CLASS_NAMES[i]) > 14:
				print("| {}\t|".format(CLASS_NAMES[i]), end="")
			else:
				print("| {}\t\t|".format(CLASS_NAMES[i]), end="")
			print("  {:.10f}\t|".format(AUROCs[i]))
		print("|---------------------------------------|")
		print("| Average\t\t|  {:.10f}\t|".format(np.array(AUROCs).mean()))
		print("|=======================================|")
		print()

def main():
	global GlobalModel, LocalModel, FusionModel

	for branch_name in BRANCH_NAME_LIST:
		print(" Start training " + branch_name + " branch...")
		batch_multiplier = exp_cfg['batch_size'][branch_name] // MAX_BATCH_CAPACITY[branch_name]

		train_dataset = ChestXrayDataSet(data_dir = IMAGES_DATA_PATH, image_list_file = TRAIN_IMAGE_LIST, transform = transform)
		train_loader = DataLoader(dataset = train_dataset, batch_size = MAX_BATCH_CAPACITY[branch_name], shuffle = True, num_workers = 4)

		val_dataset = ChestXrayDataSet(data_dir = IMAGES_DATA_PATH, image_list_file = VAL_IMAGE_LIST, transform = transform)
		val_loader = DataLoader(dataset = val_dataset, batch_size = exp_cfg['batch_size'][branch_name]//2, shuffle = False, num_workers = 4)

		if args.resume:

			CKPT_PATH = path.join(exp_dir, exp_dir.split('/')[-1] + '_'+ branch_name + '.pth')
			file_found = False

			if path.isfile(CKPT_PATH):
				save_dict = torch.load(CKPT_PATH)
				file_found = True

			if branch_name == 'global':
				GlobalModel = GlobalModel.to(device)

				if file_found:
					GlobalModel.load_state_dict(save_dict['net'])
					optimizer_global.load_state_dict(save_dict['optim'])
					lr_scheduler_global.load_state_dict(save_dict['lr_scheduler'])

			if branch_name == 'local':
				LocalModel = LocalModel.to(device)

				if file_found:
					LocalModel.load_state_dict(save_dict['net'])
					optimizer_local.load_state_dict(save_dict['optim'])
					lr_scheduler_local.load_state_dict(save_dict['lr_scheduler'])

			if branch_name == 'fusion':
				FusionModel = FusionModel.to(device)

				if file_found:
					FusionModel.load_state_dict(save_dict['net'])
					optimizer_fusion.load_state_dict(save_dict['optim'])
					lr_scheduler_fusion.load_state_dict(save_dict['lr_scheduler'])

			if file_found:
				BEST_VAL_LOSS[branch_name] = save_dict['loss_value']
				start_epoch = save_dict['epoch'] + 1
			else:
				start_epoch = 0	
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

		print(" Training " + branch_name + " branch done.")

if __name__ == "__main__":
	main()
