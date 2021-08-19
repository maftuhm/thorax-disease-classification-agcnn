import os
import os.path as path
import json
import argparse
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

from model import ResAttCheXNet, FusionNet
from utils import WeightedBCELoss, AttentionMaskInference, ChestXrayDataSet

from utils.utils import *
from config import *

def parse_args():
	parser = argparse.ArgumentParser(description='AG-CNN')
	parser.add_argument('--use', type=str, default='train', help='use for what (train or test)')
	parser.add_argument("--exp_dir", type=str, default='./final_experiments', help='define experiment directory (ex: /exp16)')
	parser.add_argument("--exp_num", type=str, default='exp0', help='define experiment directory (ex: /exp0)')
	parser.add_argument("--resume", "-r", action="store_true")
	args = parser.parse_args()
	return args

args = parse_args()

# Load config json file
with open(path.join(args.exp_dir, "cfg.json")) as f:
	config = json.load(f)

with open(path.join(args.exp_dir, "cfg_train.json")) as f:
	exp_configs = json.load(f)
	exp_config = exp_configs[args.exp_num]

if 'local' in exp_config['branch']:
	del exp_configs[exp_config['net']]['branch']

	if exp_config['net'] != '?':
		global_branch_exp = exp_config['net']
		global_branch = exp_configs[exp_config['net']]
		exp_config.update(global_branch)
	else:
		raise Exception("experiment number global branch must be choosen")

config['optimizer'] = {exp_config['optimizer'] : config['optimizer'][exp_config['optimizer']]}
del exp_config['optimizer']

config.update(exp_config)
del exp_config, exp_configs

exp_dir_num = path.join(args.exp_dir, args.exp_num)
os.makedirs(exp_dir_num, exist_ok=True)

if 'num_classes' in list(config.keys()):
	CLASS_NAMES = CLASS_NAMES[:config['num_classes']]

NUM_CLASSES = len(CLASS_NAMES)
BRANCH_NAMES = config['branch']
BEST_AUROCs = {branch: -1000 for branch in BRANCH_NAMES}
# BEST_AUROCs['global'] = 0.82879

BEST_LOSS = {branch: 1000 for branch in BRANCH_NAMES}

MAX_BATCH_CAPACITY = {
	'global' : 12,
	'local' : 6,
	'fusion' : 6
}

cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(exp_dir_num + '/log')

def train_one_epoch(epoch, branch, model, optimizer, lr_scheduler, data_loader, criterion, test_model = None):

	model.train()
	optimizer.zero_grad()

	if test_model is not None:
		for i in range(len(test_model)): test_model[i].eval()

	running_loss = 0.
	len_data = len(data_loader)
	random_int = int(torch.randint(0, len_data, (1,))[0])
	print(" Display images on index", random_int)
	batch_multiplier = config['batch_size'][branch] // MAX_BATCH_CAPACITY[branch]

	weight_last_updated = 0

	progressbar = tqdm(range(len_data))
	for i, (images, targets) in enumerate(data_loader):

		if i == random_int:
			images_draw = {}
			images_draw['images'] = images.detach()
			images_draw['targets'] = targets.detach()

		if branch == 'local':
			with torch.no_grad():
				output_global = test_model[0](images.to(device))
				output_patches = test_model[1](images.detach(), output_global['features'].detach().cpu())
				images = output_patches['crop']

		elif branch == 'fusion':
			with torch.no_grad():
				output_global = test_model[0](images.to(device))
				output_patches = test_model[1](images.detach(), output_global['features'].detach().cpu())
				output_local = test_model[2](output_patches['crop'].to(device))
				images = torch.cat((output_global['pool'], output_local['pool']), dim = 1)

		images = images.to(device)
		targets = targets.to(device)

		output = model(images)
		loss = criterion(output['out'], targets) / batch_multiplier
		loss.backward()

		if (i + 1) % batch_multiplier == 0:
			optimizer.step()
			optimizer.zero_grad()
			weight_last_updated = i + 1

		if i == random_int:
			if branch == 'global':
				draw_image = drawImage(images_draw['images'], images_draw['targets'], torch.sigmoid(output['out']).detach())
			else:
				draw_image = drawImage(images_draw['images'],
										images_draw['targets'],
										torch.sigmoid(output['out']).detach(),
										output_patches['crop'].detach(),
										output_patches['heatmap'].detach(),
										output_patches['coordinate'])

			writer.add_images("train/{}".format(branch), draw_image, epoch)

		running_loss += loss.item() * batch_multiplier

		progressbar.set_description(" Epoch: [{}/{}] | last backward: {} it | loss: {:.5f}".format(epoch, config['NUM_EPOCH'] - 1, weight_last_updated, loss.item() * batch_multiplier))
		progressbar.update(1)

	progressbar.close()

	epoch_loss = running_loss / float(len_data)
	print(' Epoch over Loss: {:.5f}'.format(epoch_loss))
	writer.add_scalars("train/loss", {branch: epoch_loss}, epoch)
	writer.add_scalars("train/learning_rate", {branch: optimizer.param_groups[0]['lr']}, epoch)

@torch.no_grad()
def val_one_epoch(epoch, branch, model, data_loader, criterion, test_model = None):

	print("\n Validating {} model".format(branch))

	model.eval()

	if test_model is not None:
		for i in range(len(test_model)): test_model[i].eval()
	
	gt = torch.FloatTensor()
	pred = torch.FloatTensor()

	running_loss = 0.
	len_data = len(data_loader)
	random_int = int(torch.randint(0, len_data, (1,))[0])
	print(" Display images on index", random_int)

	progressbar = tqdm(range(len_data))
	for i, (images, targets) in enumerate(data_loader):

		if i == random_int:
			images_draw = {}
			images_draw['images'] = images.detach()
			images_draw['targets'] = targets.detach()

		if branch == 'local':
			output_global = test_model[0](images.to(device))
			output_patches = test_model[1](images.detach(), output_global['features'].detach().cpu())
			images = output_patches['crop']

			del output_global
			torch.cuda.empty_cache()
		
		elif branch == 'fusion':
			output_global = test_model[0](images.to(device))
			output_patches = test_model[1](images.detach(), output_global['features'].detach().cpu())
			output_local = test_model[2](output_patches['crop'].to(device))
			images = torch.cat((output_global['pool'], output_local['pool']), dim = 1)

			del output_global, output_local
			torch.cuda.empty_cache()

		images = images.to(device)
		targets = targets.to(device)
		gt = torch.cat((gt, targets.detach().cpu()), 0)

		output = model(images)
		loss = criterion(output['out'], targets)
		running_loss += loss.item()

		pred = torch.cat((pred, torch.sigmoid(output['out']).detach().cpu()), 0)

		if i == random_int:
			if branch == 'global':
				draw_image = drawImage(images_draw['images'], images_draw['targets'], torch.sigmoid(output['out']).detach())
			else:
				draw_image = drawImage(images_draw['images'],
										images_draw['targets'],
										torch.sigmoid(output['out']).detach(),
										output_patches['crop'].detach(),
										output_patches['heatmap'].detach(),
										output_patches['coordinate'])

			writer.add_images("val/{}".format(branch), draw_image, epoch)

		progressbar.set_description(" Epoch: [{}/{}] | loss: {:.5f}".format(epoch,  config['NUM_EPOCH'] - 1, loss.item()))
		progressbar.update(1)

	progressbar.close()

	epoch_loss = running_loss / float(len_data)
	print(' Epoch over Loss: {:.5f}'.format(epoch_loss))
	writer.add_scalars("val/loss", {branch: epoch_loss}, epoch)

	AUROCs = compute_AUCs(gt, pred)
	AUROCs_mean = np.array(AUROCs).mean()

	writer.add_scalars("val/AUROCs", {branch: AUROCs_mean}, epoch)

	print(' Best Loss: {:.5f}'.format(BEST_LOSS[branch]))
	print("|=======================================|")
	print("|\t\t  AUROC\t\t\t|")
	print("|=======================================|")
	print("|\t      " + branch + " branch\t\t|")
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
	print("| Average\t\t|  {:.10f}\t|".format(AUROCs_mean))
	print("|=======================================|")
	print()

	return epoch_loss

def main():
	# ================= TRANSFORMS ================= #
	normalize = transforms.Normalize(
	   mean=[0.485, 0.456, 0.406],
	   std=[0.229, 0.224, 0.225]
	)

	transform_train = transforms.Compose([
	   transforms.Resize(tuple(config['dataset']['resize'])),
	   transforms.RandomResizedCrop(tuple(config['dataset']['crop'])),
	   transforms.RandomHorizontalFlip(),
	   transforms.ToTensor(),
	   normalize,
	])

	transform_test = transforms.Compose([
	   transforms.Resize(tuple(config['dataset']['resize'])),
	   transforms.CenterCrop(tuple(config['dataset']['crop'])),
	   transforms.ToTensor(),
	   normalize,
	])

	if args.resume:
		pretrained = False
	else:
		pretrained = True

	# ================= MODELS ================= #
	print("\n Model initialization")
	print(" ============================================")
	print(" Global branch")
	GlobalModel = ResAttCheXNet(pretrained = pretrained, num_classes = NUM_CLASSES, **config['net'])
	if 'local' in config['branch']:
		print(" Local branch")
		LocalModel = ResAttCheXNet(pretrained = pretrained, num_classes = NUM_CLASSES, **config['net'])
		AttentionGenPatchs = AttentionMaskInference(threshold = config['threshold'], distance_function = config['L_function'])
		print(" L distance function \t:", config['L_function'])
		print(" Threshold \t\t:", config['threshold'])
		FusionModel = FusionNet(backbone = config['net']['backbone'], num_classes = NUM_CLASSES)

	if config['loss'] == 'BCELoss':
		criterion = nn.BCELoss()
	elif config['loss'] == 'WeightedBCELoss':
		criterion = WeightedBCELoss(PosNegWeightIsDynamic = True)
	else:
		raise Exception("loss function must be BCELoss or WeightedBCELoss")

	print(" Num classes \t\t:", NUM_CLASSES)
	print(" Optimizer \t\t:", list(config['optimizer'].keys())[0])
	print(" Loss function \t\t:", config['loss'])
	print()

	for branch_name in BRANCH_NAMES:
		start_time_train = datetime.now()

		# ================= LOAD DATASET ================= #
		train_dataset = ChestXrayDataSet(data_dir = DATA_DIR, split = 'train', num_classes = NUM_CLASSES, transform = transform_train)
		train_loader = DataLoader(dataset = train_dataset, batch_size = MAX_BATCH_CAPACITY[branch_name], shuffle = True, num_workers = 4, pin_memory = True)

		val_dataset = ChestXrayDataSet(data_dir = DATA_DIR, split = 'val', num_classes = NUM_CLASSES, transform = transform_test)
		val_loader = DataLoader(dataset = val_dataset, batch_size = config['batch_size'][branch_name] // 2, shuffle = False, num_workers = 4, pin_memory = True)

		test_dataset = ChestXrayDataSet(data_dir = DATA_DIR, split = 'test', num_classes = NUM_CLASSES, transform = transform_test)
		test_loader = DataLoader(dataset = test_dataset, batch_size = config['batch_size'][branch_name] // 2, shuffle = False, num_workers = 4, pin_memory = True)

		print(" Start training " + branch_name + " branch...")
	
		if branch_name == 'global':
			Model = GlobalModel.to(device)
			TestModel = None

		if branch_name == 'local':
			save_dict_global = torch.load(os.path.join(args.exp_dir, global_branch_exp, global_branch_exp + '_global_best' + '.pth'))
			GlobalModel.load_state_dict(save_dict_global['net'])

			for param in GlobalModel.parameters():
				param.requires_grad = False

			Model = LocalModel.to(device)
			TestModel = (GlobalModel.to(device), AttentionGenPatchs)

		if branch_name == 'fusion':
			save_dict_global = torch.load(os.path.join(args.exp_dir, global_branch_exp, global_branch_exp + '_global_best' + '.pth'))
			save_dict_local = torch.load(os.path.join(exp_dir_num, args.exp_num + '_local_best' + '.pth'))

			GlobalModel.load_state_dict(save_dict_global['net'])
			LocalModel.load_state_dict(save_dict_local['net'])

			for param in GlobalModel.parameters():
				param.requires_grad = False

			for param in LocalModel.parameters():
				param.requires_grad = False

			Model = FusionModel.to(device)
			TestModel = (GlobalModel.to(device), AttentionGenPatchs, LocalModel.to(device))

		if 'SGD' in config['optimizer']:
			optimizer = optim.SGD(Model.parameters(), **config['optimizer']['SGD'])
		elif 'Adam' in config['optimizer']:
			optimizer = optim.Adam(Model.parameters(), **config['optimizer']['Adam'])
		else:
			raise Exception("optimizer must be SGD or Adam")

		# lr_scheduler = optim.lr_scheduler.StepLR(optimizer , **config['lr_scheduler'])
		lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1, verbose=True)

		if args.resume:

			checkpoint = path.join(exp_dir_num, args.exp_num + '_' + branch_name + '.pth')

			if path.isfile(checkpoint):
				save_dict = torch.load(checkpoint)
				Model.load_state_dict(save_dict['net'])
				optimizer.load_state_dict(save_dict['optim'])
				lr_scheduler.load_state_dict(save_dict['lr_scheduler'])
				BEST_LOSS[branch_name] = save_dict['loss']
				start_epoch = save_dict['epoch']
				print(" Loaded " + branch_name + " branch model checkpoint from epoch " + str(start_epoch))
				start_epoch += 1
			else:
				start_epoch = 0

		else:
			start_epoch = 0

		for epoch in range(start_epoch, config['NUM_EPOCH']):
			start_time_epoch = datetime.now()

			train_one_epoch(epoch, branch_name, Model, optimizer, lr_scheduler, train_loader, criterion, TestModel)
			
			val_loss = val_one_epoch(epoch, branch_name, Model, val_loader, criterion, TestModel)
			lr_scheduler.step(val_loss)

			save_model(exp_dir_num, epoch, val_loss, Model, optimizer, lr_scheduler, branch_name)

			if val_loss < BEST_LOSS[branch_name]:
				BEST_LOSS[branch_name] = val_loss
				save_name = os.path.join(exp_dir_num, args.exp_num + '_' + branch_name + '.pth')
				copy_name = os.path.join(exp_dir_num, args.exp_num + '_' + branch_name + '_best.pth')
				shutil.copyfile(save_name, copy_name)
				print(" Best model is saved: {}".format(copy_name))

			print(" Training epoch time: {}".format(datetime.now() - start_time_epoch))

		val_loss = val_one_epoch(config['NUM_EPOCH'], branch_name, Model, test_loader, criterion, TestModel)

		print(" Training " + branch_name + " branch done")

		print(" Training time {} branch: {}".format(branch_name, datetime.now() - start_time_train))

if __name__ == "__main__":
	main()
