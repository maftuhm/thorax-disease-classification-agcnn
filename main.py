import os
import cv2
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
# from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

from read_data import ChestXrayDataSet
from model import Net, FusionNet

cudnn.benchmark = True

def parse_args():
	parser = argparse.ArgumentParser(description='AG-CNN')
	parser.add_argument('--use', type=str, default='train', help='use for what (train or test)')
	parser.add_argument("--exp_dir", type=str, default="./experiments/exp5")
	parser.add_argument("--resume", "-r", action="store_true")
	args = parser.parse_args()
	return args
args = parse_args()

exp_dir = args.exp_dir
exp_name = exp_dir.split('/')[-1]

with open(path.join(exp_dir, "cfg.json")) as f:
	exp_cfg = json.load(f)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# writer = SummaryWriter(exp_dir + '/log')

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

transform_train = transforms.Compose([
   transforms.Resize(tuple(exp_cfg['dataset']['resize'])),
   transforms.RandomResizedCrop(tuple(exp_cfg['dataset']['crop'])),
   transforms.RandomHorizontalFlip(),
   transforms.ToTensor(),
   normalize,
])

transform_test = transforms.Compose([
   transforms.Resize(tuple(exp_cfg['dataset']['resize'])),
   transforms.CenterCrop(tuple(exp_cfg['dataset']['crop'])),
   transforms.ToTensor(),
   normalize,
])


CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
				'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

DATASET_PATH = path.join('..', 'lung-disease-detection', 'data')
IMAGES_DATA_PATH = path.join(DATASET_PATH, 'images')
TRAIN_IMAGE_LIST = path.join(DATASET_PATH, 'labels', 'train_list.txt')
VAL_IMAGE_LIST = path.join(DATASET_PATH, 'labels', 'val_list.txt')
TEST_IMAGE_LIST = path.join(DATASET_PATH, 'labels', 'test_list.txt')

MAX_BATCH_CAPACITY = 2

train_dataset = ChestXrayDataSet(data_dir = IMAGES_DATA_PATH, image_list_file = TRAIN_IMAGE_LIST, transform = transform_train)
train_loader = DataLoader(dataset = train_dataset, batch_size = MAX_BATCH_CAPACITY, shuffle = True, num_workers = 4)

val_dataset = ChestXrayDataSet(data_dir = IMAGES_DATA_PATH, image_list_file = VAL_IMAGE_LIST, transform = transform_test)
val_loader = DataLoader(dataset = val_dataset, batch_size = 64, shuffle = False, num_workers = 4)

test_dataset = ChestXrayDataSet(data_dir = IMAGES_DATA_PATH, image_list_file = TEST_IMAGE_LIST, transform = transform_test)
test_loader = DataLoader(dataset = test_dataset, batch_size = 64, shuffle = False, num_workers = 4)


BRANCH_NAME_LIST = ['global', 'local', 'fusion']
BEST_VAL_LOSS = {
	'global': 1000,
	'local': 1000,
	'fusion': 1000
}

#-------------------- SETTINGS: MODEL
GlobalModel = Net(exp_cfg['backbone']).to(device)
LocalModel = Net(exp_cfg['backbone']).to(device)
FusionModel = FusionNet(exp_cfg['backbone']).to(device)

#-------------------- SETTINGS: OPTIMIZER & SCHEDULER
optimizer_global = optim.SGD(GlobalModel.parameters(), **exp_cfg['optimizer']['SGD'])
optimizer_local = optim.SGD(LocalModel.parameters(), **exp_cfg['optimizer']['SGD'])
optimizer_fusion = optim.SGD(FusionModel.parameters(), **exp_cfg['optimizer']['SGD'])

lr_scheduler_global = optim.lr_scheduler.StepLR(optimizer_global , **exp_cfg['lr_scheduler'])
lr_scheduler_local = optim.lr_scheduler.StepLR(optimizer_local , **exp_cfg['lr_scheduler'])
lr_scheduler_fusion = optim.lr_scheduler.StepLR(optimizer_fusion , **exp_cfg['lr_scheduler'])

#-------------------- SETTINGS: LOSS FUNCTION
criterion = nn.BCELoss()

def AttentionGenPatchs(ori_image, fm_cuda):

	feature_conv = fm_cuda.data.cpu().numpy()

	bz, nc, h, w = feature_conv.shape

	images = torch.randn(bz, 3, 224, 224)

	for i in range(bz):
		feature = np.abs(feature_conv[i])
		heatmap = np.max(feature, axis = 0)
		heatmap = (heatmap - np.min(heatmap)) / np.max(heatmap)
		heatmap = np.uint8(255 * heatmap)

		resized_heatmap = cv2.resize(heatmap, (224, 224))
		_, heatmap_bin = cv2.threshold(resized_heatmap , 0 , 255 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		# _, heatmap_bin = cv2.threshold(resized_heatmap , int(255 * 0.7) , 255 , cv2.THRESH_BINARY)
		heatmap_maxconn = selectMaxConnect(heatmap_bin)
		heatmap_mask = heatmap_bin * heatmap_maxconn

		ind = np.argwhere(heatmap_mask != 0)
		minh = min(ind[:,0])
		minw = min(ind[:,1])
		maxh = max(ind[:,0])
		maxw = max(ind[:,1])

		image = ori_image[i].numpy().transpose(1, 2, 0)
		image_crop = image[minh:maxh, minw:maxw, :]
		image_crop = cv2.resize(image_crop, (224, 224))
		image_crop = Image.fromarray(image_crop.astype('uint8')).convert("RGB")
		images[i] = transforms.ToTensor()(image_crop)

	return images

def selectMaxConnect(heatmap):
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

def train(epoch):
	
	GlobalModel.train()
	LocalModel.train()
	FusionModel.train()
	
	running_loss = 0.
	mini_batch_loss = 0.

	progressbar = tqdm(range(len(train_loader)))

	for i, (image, target) in enumerate(train_loader):

		image_cuda = image.to(device)
		target_cuda = target.to(device)

		optimizer_global.zero_grad()
		optimizer_local.zero_grad()
		optimizer_fusion.zero_grad()

		# compute output
		output_global, fm_global, pool_global = GlobalModel(image_cuda)
		
		image_patch = AttentionGenPatchs(image_cuda.cpu(), fm_global).to(device)

		output_local, _, pool_local = LocalModel(image_patch)

		output_fusion = FusionModel(pool_global, pool_local)

		# loss
		loss_global = criterion(output_global, target_cuda)
		loss_local = criterion(output_local, target_cuda)
		loss_fusion = criterion(output_fusion, target_cuda)

		loss = loss_global + loss_local + loss_fusion
		loss.backward()

		optimizer_global.step()
		optimizer_local.step()
		optimizer_fusion.step()
		
		progressbar.set_description(" Epoch: [{}/{}] | loss: {:.5f}".format(epoch, exp_cfg['NUM_EPOCH'] - 1, loss.data.item()))
		progressbar.update(1)

		# if count % batch_multiplier == 0:
		# 	writer.add_scalar('train/' + branch_name + ' loss', mini_batch_loss, epoch * len(data_loader) + i)
		# 	mini_batch_loss = 0.

		running_loss += loss.data.item()

	progressbar.close()

	lr_scheduler_global.step()
	lr_scheduler_local.step() 
	lr_scheduler_fusion.step() 

	epoch_train_loss = float(running_loss) / float(i)
	print(' Epoch over Loss: {:.5f}'.format(epoch_train_loss))

	# save_model(epoch, epoch_train_loss, model, optimizer, lr_scheduler, branch_name)
	print()

	# del image, target, loss, epoch_train_loss, output, heatmap, pool, model
	# torch.cuda.empty_cache()

def val(epoch, data_loader):

	GlobalModel.eval()
	LocalModel.eval()
	FusionModel.eval()

	ground_truth = torch.FloatTensor()
	pred_global = torch.FloatTensor()
	pred_local = torch.FloatTensor()
	pred_fusion = torch.FloatTensor()

	running_loss = 0.
	progressbar = tqdm(range(len(data_loader)))

	with torch.no_grad():
		for i, (image, target) in enumerate(data_loader):

			image_cuda = image.to(device)
			target_cuda = target.to(device)
			ground_truth = torch.cat((ground_truth, target), 0)

			# compute output
			output_global, fm_global, pool_global = GlobalModel(image_cuda)
			
			image_patch = AttentionGenPatchs(image_cuda.cpu(), fm_global).to(device)

			output_local, _, pool_local = LocalModel(image_patch)

			output_fusion = FusionModel(pool_global, pool_local)

			# loss
			loss_global = criterion(output_global, target_cuda)
			loss_local = criterion(output_local, target_cuda)
			loss_fusion = criterion(output_fusion, target_cuda)

			loss = loss_global + loss_local + loss_fusion

			
			pred_global = torch.cat((pred_global, output_global.data), 0)
			pred_local = torch.cat((pred_local, output_local.data), 0)
			pred_fusion = torch.cat((pred_fusion, output_fusion.data), 0)

			running_loss += loss.data.item()

			progressbar.set_description(" Epoch: [{}/{}] | loss: {:.5f}".format(epoch,  exp_cfg['NUM_EPOCH'] - 1, loss.data.item()))
			progressbar.update(1)
			# writer.add_scalar('val/' + branch_name + ' loss', loss.data.item(), epoch * len(data_loader) + i)

		progressbar.close()
		
		epoch_val_loss = float(running_loss) / float(i)
		print(' Epoch over Loss: {:.5f}'.format(epoch_val_loss))

		if args.use == 'train':
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

	print(" Start training branch...")

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

	if args.use == 'train':
		max_epoch = exp_cfg['NUM_EPOCH']
	elif args.use == 'test':
		max_epoch = 1

	for epoch in range(start_epoch, max_epoch):
		train(epoch)
		val(epoch, val_loader)

	print(" Training branch done.")

# if __name__ == "__main__":
	# main()
