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

# import torchvision.transforms as transforms

from model import MainNet, FusionNet
from utils import WeightedBCELoss, AttentionMaskInference, ChestXrayDataSet
from utils import transforms

from utils.utils import *
from config import *

def parse_args():
	parser = argparse.ArgumentParser(description='AG-CNN')
	parser.add_argument('--use', type=str, default='train', help='use for what (train or test)')
	parser.add_argument("--exp_dir", type=str, default='./experiments2', help='define experiment directory (ex: /exp16)')
	parser.add_argument("--exp_num", type=str, default='exp0', help='define experiment directory (ex: /exp0)')
	parser.add_argument("--resume", "-r", action="store_true")
	args = parser.parse_args()
	return args

def memory_usage(info = ""):
	print("\n ON " + info)
	print(" memory_allocated:", torch.cuda.memory_allocated() / 1024**2)
	print(" max_memory_allocated:", torch.cuda.max_memory_allocated() / 1024**2)
	print(" memory_reserved:", torch.cuda.memory_reserved() / 1024**2)
	print(" max_memory_reserved:", torch.cuda.max_memory_reserved() / 1024**2)

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

config['lr_scheduler'] = {exp_config['lr_scheduler'] : config['lr_scheduler'][exp_config['lr_scheduler']]}
del exp_config['lr_scheduler']

config.update(exp_config)
del exp_config, exp_configs

exp_dir_num = path.join(args.exp_dir, args.exp_num)
os.makedirs(exp_dir_num, exist_ok=True)

if 'num_classes' in list(config.keys()):
	CLASS_NAMES = CLASS_NAMES[:config['num_classes']]

NUM_CLASSES = len(CLASS_NAMES)
BRANCH_NAMES = config['branch']
BEST_AUROCs = {branch: 0. for branch in BRANCH_NAMES}
# BEST_AUROCs['global'] = 0.82966

BEST_LOSS = {branch: 1000. for branch in BRANCH_NAMES}
# BEST_LOSS['global'] = 0.13489

MAX_BATCH_CAPACITY = config['max_batch']

cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(exp_dir_num + '/log')

def train_one_epoch(epoch, branch, model, optimizer, lr_scheduler, data_loader, criterion, test_model = None):

	model.train()

	# if test_model is not None:
	# 	for key in test_model:
	# 		if key != 'attention':
	# 			test_model[key].train()

	optimizer.zero_grad()

	running_loss = 0.
	len_data = len(data_loader)
	random_int = torch.randint(0, len_data, (1,)).item()
	print(" Display images on index", random_int)
	batch_multiplier = config['batch_size'][branch] // MAX_BATCH_CAPACITY[branch]

	weight_last_updated = 0

	progressbar = tqdm(range(len_data))
	for i, (images, targets) in enumerate(data_loader):

		if (i + 1) > (len_data - len_data % batch_multiplier):
			batch_multiplier = len_data % batch_multiplier

		if i == random_int:
			images_draw = {}
			images_draw['images'] = images.detach()
			images_draw['targets'] = targets.detach()

		if branch == 'local':
			with torch.no_grad():
				output_global = test_model['global'](images.to(device))
				output_patches = test_model['attention'](images.detach(), output_global['features'].detach().cpu())
				images = output_patches['image']
				del output_global
				torch.cuda.empty_cache()

		elif branch == 'fusion':
			with torch.no_grad():
				output_global = test_model['global'](images.to(device))
				output_patches = test_model['attention'](images.detach(), output_global['features'].detach().cpu())
				output_local = test_model['local'](output_patches['image'].to(device))
				images = torch.cat((output_global['pool'], output_local['pool']), dim = 1)

				del output_global, output_local
				torch.cuda.empty_cache()

		images = images.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)

		# be careful add last layer with sigmoid if using bceloss or weighted bce loss
		# and remove sigimoid(output) here
		output = model(images)
		loss = criterion(output['score'], targets) / batch_multiplier
		loss.backward()

		if (i + 1) % batch_multiplier == 0:
			optimizer.step()
			optimizer.zero_grad()
			weight_last_updated = i + 1

		if i == random_int:
			if branch == 'global':
				output_patches = test_model['attention'](images_draw['images'], output['features'].detach().cpu())

			draw_image = drawImage(images_draw['images'],
									images_draw['targets'],
									output['score'].detach().cpu(),
									output_patches['image'].detach().cpu(),
									output_patches['heatmap'].detach().cpu(),
									None,
									output_patches['coordinate'])

			writer.add_images("train/{}".format(branch), draw_image, epoch)

			del images_draw, draw_image

		running_loss += loss.item() * batch_multiplier

		progressbar.set_description(" Epoch: [{}/{}] | last backward: {} it | loss: {:.5f}".format(epoch, config['NUM_EPOCH'] - 1, weight_last_updated, loss.item() * batch_multiplier))
		progressbar.update(1)

	progressbar.close()

	epoch_loss = running_loss / float(len_data)
	print(' Epoch over Loss: {:.5f}'.format(epoch_loss))
	writer.add_scalars("train/loss", {branch: epoch_loss}, epoch)
	writer.add_scalars("train/learning_rate", {branch: optimizer.param_groups[0]['lr']}, epoch)
	del images, targets, output, running_loss, epoch_loss
	torch.cuda.empty_cache()

@torch.no_grad()
def val_one_epoch(epoch, branch, model, data_loader, criterion, test_model = None):

	print("\n Validating {} model".format(branch))

	model.eval()

	# if test_model is not None:
	# 	for key in test_model:
	# 		if key != 'attention':
	# 			test_model[key].eval()
	
	gt = torch.FloatTensor()
	pred = torch.FloatTensor()

	running_loss = 0.
	len_data = len(data_loader)
	random_int = torch.randint(0, len_data, (1,)).item()
	print(" Display images on index", random_int)

	progressbar = tqdm(range(len_data))
	for i, (images, targets) in enumerate(data_loader):

		if i == random_int:
			images_draw = {}
			images_draw['images'] = images.detach()
			images_draw['targets'] = targets.detach()

		if branch == 'local':
			output_global = test_model['global'](images.to(device))
			output_patches = test_model['attention'](images.detach(), output_global['features'].detach().cpu())
			images = output_patches['image']

			del output_global
			torch.cuda.empty_cache()

		elif branch == 'fusion':
			output_global = test_model['global'](images.to(device))
			output_patches = test_model['attention'](images.detach(), output_global['features'].detach().cpu())
			output_local = test_model['local'](output_patches['image'].to(device))
			images = torch.cat((output_global['pool'], output_local['pool']), dim = 1)

			del output_global, output_local
			torch.cuda.empty_cache()

		images = images.to(device)
		targets = targets.to(device)
		gt = torch.cat((gt, targets.detach().cpu()), 0)

		# be careful add last layer with sigmoid if using bceloss or weighted bce loss
		# and remove sigimoid(output) here
		output = model(images)
		loss = criterion(output['score'], targets)
		running_loss += loss.item()

		pred = torch.cat((pred, output['score'].detach().cpu()), 0)

		if i == random_int:

			if branch == 'global':
				output_patches = test_model['attention'](images_draw['images'], output['features'].detach().cpu())

			draw_image = drawImage(images_draw['images'],
									images_draw['targets'],
									output['score'].detach(),
									output_patches['image'].detach().cpu(),
									output_patches['heatmap'].detach().cpu(),
									None,
									output_patches['coordinate'])

			writer.add_images("val/{}".format(branch), draw_image, epoch)
			del images_draw, draw_image

		progressbar.set_description(" Epoch: [{}/{}] | loss: {:.5f}".format(epoch,  config['NUM_EPOCH'] - 1, loss.item()))
		progressbar.update(1)

	progressbar.close()

	epoch_loss = running_loss / float(len_data)
	print(' Epoch over Loss: {:.5f}'.format(epoch_loss))
	writer.add_scalars("val/loss", {branch: epoch_loss}, epoch)

	AUROCs = compute_AUCs(gt, pred)
	if len(CLASS_NAMES) > 14:
		AUROCs_mean = np.array(AUROCs)[:14].mean()
	else:
		AUROCs_mean = np.array(AUROCs).mean()

	writer.add_scalars("val/AUROCs", {branch: AUROCs_mean}, epoch)

	print(' Best AUROCs: {:.5f} | Best Loss: {:.5f}'.format(BEST_AUROCs[branch], BEST_LOSS[branch]))
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

	del images, targets, output, running_loss
	torch.cuda.empty_cache()

	return AUROCs_mean, epoch_loss

def main():
	# ================= TRANSFORMS ================= #

	# normalize = transforms.Normalize(
	#    mean=torch.tensor([[0.485, 0.456, 0.406]]).mean(1),
	#    std=torch.tensor([[0.229, 0.224, 0.225]]).mean(1)
	# )

	transform_init = transforms.Resize(tuple(config['dataset']['resize']))
	transform_train = transforms.Compose(
	   transforms.RandomResizedCrop(tuple(config['dataset']['crop']), (0.25, 1.0)),
	   transforms.RandomHorizontalFlip(),
	   transforms.ToTensor(),
	   # transforms.Normalize(mean=[0.49886124425113754], std=[0.22925289787072856])
	   transforms.DynamicNormalize()
	)

	transform_test = transforms.Compose(
	   transforms.CenterCrop(tuple(config['dataset']['crop'])),
	   transforms.ToTensor(),
	   # transforms.Normalize(mean=[0.49886124425113754], std=[0.22925289787072856])
	   transforms.DynamicNormalize()
	)

	if args.resume:
		config['net'].update({"pretrained" : False})

	# ================= MODELS ================= #
	print("\n Model initialization")
	print(" ============================================")
	print(" Global branch")
	GlobalModel = MainNet(num_classes = NUM_CLASSES, **config['net'])
	if 'local' in config['branch']:
		print(" Local branch")
		LocalModel = MainNet(num_classes = NUM_CLASSES, **config['net'])
		AttentionGenPatchs = AttentionMaskInference(threshold = config['threshold'], distance_function = config['L_function'])
		print(" L distance function \t:", config['L_function'])
		print(" Threshold \t\t:", config['threshold'])
		FusionModel = FusionNet(threshold = config['threshold'], distance_function = config['L_function'], num_classes = NUM_CLASSES, **config['net'])

	print(" Num classes \t\t:", NUM_CLASSES)
	print(" Optimizer \t\t:", list(config['optimizer'].keys())[0])
	print(" Lr Scheduler \t\t:", list(config['lr_scheduler'].keys())[0])
	print(" Loss function \t\t:", config['loss'])
	print()

	for branch_name in BRANCH_NAMES:
		start_time_train = datetime.now()

		# ================= LOAD DATASET ================= #
		train_dataset = ChestXrayDataSet(DATA_DIR, 'train', num_classes = NUM_CLASSES, transform = transform_train, init_transform=transform_init)
		train_loader = DataLoader(dataset = train_dataset, batch_size = MAX_BATCH_CAPACITY[branch_name], shuffle = True, num_workers = 5, pin_memory = True, drop_last=True)

		val_dataset = ChestXrayDataSet(DATA_DIR, 'val', num_classes = NUM_CLASSES, transform = transform_test, init_transform=transform_init)
		val_loader = DataLoader(dataset = val_dataset, batch_size = 96, shuffle = False, num_workers = 5, pin_memory = False)

		test_dataset = ChestXrayDataSet(DATA_DIR, 'test', num_classes = NUM_CLASSES, transform = transform_test, init_transform=transform_init)
		test_loader = DataLoader(dataset = test_dataset, batch_size = 96, shuffle = False, num_workers = 5, pin_memory = False)

		if config['loss'] == 'BCELoss':
			criterion = nn.BCELoss()
		elif config['loss'] == 'WeightedBCELoss':
			criterion = WeightedBCELoss(PosNegWeightIsDynamic = True)
			# criterion = dict(
			# 	train = WeightedBCELoss(weight = torch.tensor(0.1), pos_weight = get_weight_wbce_loss(train_dataset.labels)),
			# 	val = WeightedBCELoss(weight = torch.tensor(0.1), pos_weight = get_weight_wbce_loss(val_dataset.labels)),
			# 	test = WeightedBCELoss(weight = torch.tensor(0.1), pos_weight = get_weight_wbce_loss(test_dataset.labels))
			# )
		elif config['loss'] == 'BCEWithLogitsLoss':
			criterion = dict(
				train = nn.BCEWithLogitsLoss(weight = torch.tensor(0.1), pos_weight = get_weight_wbce_loss(train_dataset.labels)),
				val = nn.BCEWithLogitsLoss(weight = torch.tensor(0.1), pos_weight = get_weight_wbce_loss(val_dataset.labels)),
				test = nn.BCEWithLogitsLoss(weight = torch.tensor(0.1), pos_weight = get_weight_wbce_loss(test_dataset.labels))
			)
		else:
			raise Exception("loss function must be BCELoss or WeightedBCELoss")

		print(" Start training " + branch_name + " branch...")
	
		if branch_name == 'global':
			Model = GlobalModel.to(device)
			TestModel = {
				'attention': AttentionGenPatchs
			}

		if branch_name == 'local':
			save_dict_global = torch.load(os.path.join(args.exp_dir, global_branch_exp, global_branch_exp + '_global_best_auroc' + '.pth'))
			GlobalModel.load_state_dict(save_dict_global['net'])

			for param in GlobalModel.parameters():
				param.requires_grad = False

			Model = LocalModel.to(device)
			TestModel = {
				'global' : GlobalModel.to(device),
				'attention': AttentionGenPatchs
			}

			for key in TestModel: 
				if key != 'attention':
					TestModel[key].eval()

			del save_dict_global
			torch.cuda.empty_cache()

		if branch_name == 'fusion':
			save_dict_global = torch.load(os.path.join(args.exp_dir, global_branch_exp, global_branch_exp + '_global_best_auroc' + '.pth'), map_location='cpu')
			save_dict_local = torch.load(os.path.join(exp_dir_num, args.exp_num + '_local_best_auroc' + '.pth'), map_location='cpu')

			GlobalModel.load_state_dict(save_dict_global['net'])
			LocalModel.load_state_dict(save_dict_local['net'])

			# for param in GlobalModel.parameters():
			# 	param.requires_grad = False

			# for param in LocalModel.parameters():
			# 	param.requires_grad = False

			# FusionModel.load_branch_weight(save_dict_global['net'], save_dict_local['net'])

			Model = FusionModel.to(device)
			TestModel = None
			TestModel = {
				'global' : GlobalModel.to(device), 
				'attention' : AttentionGenPatchs, 
				'local' : LocalModel.to(device)
			}

			for key in TestModel: 
				if key != 'attention':
					TestModel[key].eval()

			del save_dict_global, save_dict_local
			torch.cuda.empty_cache()

			# for op in config['optimizer']:
			# 	config['optimizer'][op]['lr'] /= 10

		if 'SGD' in config['optimizer']:
			optimizer = optim.SGD(Model.parameters(), **config['optimizer']['SGD'])
		elif 'Adam' in config['optimizer']:
			optimizer = optim.Adam(Model.parameters(), **config['optimizer']['Adam'])
		else:
			raise Exception("optimizer must be SGD or Adam")

		if 'StepLR' in config['lr_scheduler']:
			lr_scheduler = optim.lr_scheduler.StepLR(optimizer , **config['lr_scheduler']['StepLR'])
		elif 'ReduceLROnPlateau' in config['lr_scheduler']:
			lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config['lr_scheduler']['ReduceLROnPlateau'])
		else:
			raise Exception("lr_scheduler must be StepLR or ReduceLROnPlateau")

		if args.resume:

			checkpoint = path.join(exp_dir_num, args.exp_num + '_' + branch_name + '.pth')

			if path.isfile(checkpoint):

				save_dict = torch.load(checkpoint)
				Model.load_state_dict(save_dict['net'])
				Model = Model.to(device)
				optimizer.load_state_dict(save_dict['optim'])
				lr_scheduler.load_state_dict(save_dict['lr_scheduler'])
				start_epoch = save_dict['epoch']
				print(" Loaded " + branch_name + " branch model checkpoint from epoch " + str(start_epoch))
				start_epoch += 1

				del save_dict
				torch.cuda.empty_cache()
			#else:
			#	raise Exception("checkpoint model does not exist")

			checkpoint_best_auroc = path.join(exp_dir_num, args.exp_num + '_' + branch_name + '_best_auroc.pth')
			checkpoint_best_loss = path.join(exp_dir_num, args.exp_num + '_' + branch_name + '_best_loss.pth')

			if path.isfile(checkpoint_best_auroc) and path.isfile(checkpoint_best_loss):
				save_dict_best_loss = torch.load(checkpoint_best_loss)
				save_dict_best_auroc = torch.load(checkpoint_best_auroc)
				BEST_LOSS[branch_name] = save_dict_best_loss.get('loss', 1000.)
				BEST_AUROCs[branch_name] = save_dict_best_auroc.get('auroc', 0.)
				print(" latest best loss:", BEST_LOSS[branch_name])
				print(" latest best auroc:", BEST_AUROCs[branch_name])
				del save_dict_best_loss, save_dict_best_auroc
				torch.cuda.empty_cache()

		else:
			start_epoch = 0

		for epoch in range(start_epoch, config['NUM_EPOCH']):
			start_time_epoch = datetime.now()

			train_one_epoch(epoch, branch_name, Model, optimizer, lr_scheduler, train_loader, criterion['train'].to(device) if isinstance(criterion, dict) else criterion, TestModel)

			val_auroc, val_loss = val_one_epoch(epoch, branch_name, Model, val_loader, criterion['val'].to(device) if isinstance(criterion, dict) else criterion, TestModel)

			if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
				distance_loss_auroc = torch.tensor([val_auroc, val_loss]) ** 2
				distance_loss_auroc = distance_loss_auroc.sum().sqrt()
				writer.add_scalars("val/distance_loss_auroc", {branch_name: distance_loss_auroc}, epoch)

				lr_scheduler.step(distance_loss_auroc)
			else:
				lr_scheduler.step()

			save_model(exp_dir_num, epoch, val_auroc, val_loss, Model, optimizer, lr_scheduler, branch_name)

			save_name = os.path.join(exp_dir_num, args.exp_num + '_' + branch_name + '.pth')
			if val_auroc > BEST_AUROCs[branch_name]:
				BEST_AUROCs[branch_name] = val_auroc
				copy_name = os.path.join(exp_dir_num, args.exp_num + '_' + branch_name + '_best_auroc.pth')
				shutil.copyfile(save_name, copy_name)
				print(" Best model based on AUROCs is saved: {}".format(copy_name))

			if val_loss < BEST_LOSS[branch_name]:
				BEST_LOSS[branch_name] = val_loss
				copy_name = os.path.join(exp_dir_num, args.exp_num + '_' + branch_name + '_best_loss.pth')
				shutil.copyfile(save_name, copy_name)
				print(" Best model based on loss is saved: {}".format(copy_name))

			print(" Training epoch time: {}\n".format(datetime.now() - start_time_epoch))

		# val_one_epoch(config['NUM_EPOCH'], branch_name, Model, test_loader, criterion['test'].to(device) if isinstance(criterion, dict) else criterion, TestModel)
		del Model, TestModel, train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader, criterion, optimizer
		torch.cuda.empty_cache()

		print(" Training " + branch_name + " branch done")
		start_epoch = 0

		print(" Training time {} branch: {}".format(branch_name, datetime.now() - start_time_train))

if __name__ == "__main__":
	main()
