import os
import os.path as path
import json
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

from read_data import ChestXrayDataSet
from model import Net, FusionNet
from utils import *

def parse_args():
	parser = argparse.ArgumentParser(description='AG-CNN')
	parser.add_argument('--use', type=str, default='train', help='use for what (train or test)')
	parser.add_argument("--exp_dir", type=str, default="./experiments/exp5")
	parser.add_argument("--resume", "-r", action="store_true")
	args = parser.parse_args()
	return args

args = parse_args()

# Load config json file
with open(path.join(args.exp_dir, "cfg.json")) as f:
	exp_cfg = json.load(f)

# ================= CONSTANTS ================= #
data_dir = path.join('..', 'lung-disease-detection', 'data')
classes_name = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
				'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

max_batch_capacity = 8

best_loss = {
	'global': 1000,
	'local': 1000,
	'fusion': 1000
}

cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# writer = SummaryWriter(args.exp_dir + '/log')

def main():
	# ================= TRANSFORMS ================= #
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

	# ================= LOAD DATASET ================= #
	train_dataset = ChestXrayDataSet(data_dir = data_dir,split = 'train', transform = transform_train)
	train_loader = DataLoader(dataset = train_dataset, batch_size = max_batch_capacity, shuffle = True, num_workers = 4)

	val_dataset = ChestXrayDataSet(data_dir = data_dir, split = 'val', transform = transform_test)
	val_loader = DataLoader(dataset = val_dataset, batch_size = 64, shuffle = False, num_workers = 4)

	test_dataset = ChestXrayDataSet(data_dir = data_dir, split = 'test', transform = transform_test)
	test_loader = DataLoader(dataset = test_dataset, batch_size = 64, shuffle = False, num_workers = 4)

	# ================= MODELS ================= #
	GlobalModel = Net(exp_cfg['backbone'])
	LocalModel = Net(exp_cfg['backbone'])
	FusionModel = FusionNet(exp_cfg['backbone'])

	# ================= OPTIMIZER ================= #
	optimizer_global = optim.SGD(GlobalModel.parameters(), **exp_cfg['optimizer']['SGD'])
	optimizer_local = optim.SGD(LocalModel.parameters(), **exp_cfg['optimizer']['SGD'])
	optimizer_fusion = optim.SGD(FusionModel.parameters(), **exp_cfg['optimizer']['SGD'])

	# ================= SCHEDULER ================= #
	lr_scheduler_global = optim.lr_scheduler.StepLR(optimizer_global , **exp_cfg['lr_scheduler'])
	lr_scheduler_local = optim.lr_scheduler.StepLR(optimizer_local , **exp_cfg['lr_scheduler'])
	lr_scheduler_fusion = optim.lr_scheduler.StepLR(optimizer_fusion , **exp_cfg['lr_scheduler'])

	# ================= LOSS FUNCTION ================= #
	criterion = nn.BCELoss()

	if args.resume:
		start_epoch = 0
		checkpoint_global = path.join(args.exp_dir, args.exp_dir.split('/')[-1] + '_global.pth')
		checkpoint_local = path.join(args.exp_dir, args.exp_dir.split('/')[-1] + '_local.pth')
		checkpoint_fusion = path.join(args.exp_dir, args.exp_dir.split('/')[-1] + '_fusion.pth')

		if path.isfile(checkpoint_global):
			save_dict = torch.load(checkpoint_global)
			start_epoch = max(save_dict['epoch'], start_epoch)
			best_loss['global'] = save_dict['loss']
			GlobalModel.load_state_dict(save_dict['net'])
			optimizer_global.load_state_dict(save_dict['optim'])
			lr_scheduler_global.load_state_dict(save_dict['lr_scheduler'])
			print("Loaded Global Branch Model checkpoint")

		if path.isfile(checkpoint_local):
			save_dict = torch.load(checkpoint_local)
			start_epoch = max(save_dict['epoch'], start_epoch)
			best_loss['local'] = save_dict['loss']
			LocalModel.load_state_dict(save_dict['net'])
			optimizer_local.load_state_dict(save_dict['optim'])
			lr_scheduler_local.load_state_dict(save_dict['lr_scheduler'])
			print("Loaded Local Branch Model checkpoint")

		if path.isfile(checkpoint_fusion):
			save_dict = torch.load(checkpoint_fusion)
			start_epoch = max(save_dict['epoch'], start_epoch)
			best_loss['fusion'] = save_dict['loss']
			FusionModel.load_state_dict(save_dict['net'])
			optimizer_fusion.load_state_dict(save_dict['optim'])
			lr_scheduler_fusion.load_state_dict(save_dict['lr_scheduler'])
			print("Loaded Fusion Model checkpoint")

		start_epoch += 1

	else:
		start_epoch = 0

	GlobalModel = GlobalModel.to(device)
	LocalModel = LocalModel.to(device)
	FusionModel = FusionModel.to(device)

	for epoch in range(start_epoch, exp_cfg['NUM_EPOCH']):
		print('Epoch [{}/{}]\tTotal Iterasi [{}]'.format(epoch , exp_cfg['NUM_EPOCH'] - 1, len(train_loader)))

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
			
			progressbar.set_description("bacth loss: {loss:.5f} "
										"loss1: {loss1:.3f} "
										"loss2: {loss2:.3f} "
										"loss3: {loss3:.3f}".format(loss = loss,
																	loss1 = loss_global,
																	loss2 = loss_local,
																	loss3 = loss_fusion))
			progressbar.update(1)

			running_loss += loss.data.item()

		progressbar.close()

		lr_scheduler_global.step()
		lr_scheduler_local.step()
		lr_scheduler_fusion.step()

		epoch_train_loss = float(running_loss) / float(i)
		print(' Epoch over Loss: {:.5f}'.format(epoch_train_loss))

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
				save_name = path.join(args.exp_dir, args.exp_dir.split('/')[-1] + '_' + branch_name + '.pth')
				copy_name = os.path.join(args.exp_dir, args.exp_dir.split('/')[-1] + '_' + branch_name + '_best.pth')
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

if __name__ == "__main__":
	main()
