import os
import os.path as path
import json
import argparse
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

from read_data import ChestXrayDataSet
from model import ResAttCheXNet, FusionNet, WeightedBCELoss
from utils import *

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

exp_dir_num = path.join(args.exp_dir, args.exp_num)
os.makedirs(exp_dir_num, exist_ok=True)


# ================= CONSTANTS ================= #
data_dir = path.join('..', 'lung-disease-detection', 'data')

BRANCH_NAMES = exp_config['branch']
BEST_AUROCs = {branch: -1000 for branch in BRANCH_NAMES}

MAX_BATCH_CAPACITY = {
	'global' : 20,
	'local' : 10,
	'fusion' : 10
}

cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(exp_dir_num + '/log')

def train_one_epoch(epoch, branch, model, optimizer, lr_scheduler, data_loader, criterion, test_model = None):

	model.train()
	optimizer.zero_grad()

	running_loss = 0.
	len_data = len(data_loader)
	random_int = int(torch.randint(0, len_data, (1,))[0])
	print(" display images on index", random_int)
	batch_multiplier = config['batch_size'][branch] // MAX_BATCH_CAPACITY[branch]

	progressbar = tqdm(range(len_data))
	for i, (images, targets) in enumerate(data_loader):

		if i == random_int:
			images_draw = {}
			images_draw['images'] = images.data
			images_draw['targets'] = targets.data

		if branch == 'local':
			with torch.no_grad():
				output_global = test_model(images.to(device))
				output_patches = AttentionGenPatchs(images.data, output_global['features'].data.cpu(), exp_config['threshold'], exp_config['L_function'])
				images = output_patches['crop']

			del output_global
			torch.cuda.empty_cache()
		
		elif branch == 'fusion':
			with torch.no_grad():
				output_global = test_model[0](images.to(device))
				output_patches = AttentionGenPatchs(images.data, output_global['features'].data.cpu(), exp_config['threshold'], exp_config['L_function'])
				output_local = test_model[1](output_patches['crop'].to(device))
				images = torch.cat((output_global['pool'], output_local['pool']), dim = 1)

			del output_global, output_local
			torch.cuda.empty_cache()

		images = images.to(device)
		targets = targets.to(device)

		output = model(images)

		loss = criterion(output['out'], targets) / batch_multiplier
		running_loss += loss.data.item() * batch_multiplier

		loss.backward()
		if (i + 1) % batch_multiplier == 0:
			optimizer.step()
			optimizer.zero_grad()

		if i == random_int:
			if branch == 'global':
				draw_image = drawImage(images_draw['images'], images_draw['targets'], torch.sigmoid(output['out']).data)
			else:
				draw_image = drawImage(images_draw['images'],
										images_draw['targets'],
										torch.sigmoid(output['out']).data,
										output_patches['crop'].data,
										output_patches['heatmap'].data,
										output_patches['coordinate'])

			writer.add_images("train/{}".format(branch), draw_image, epoch)

		progressbar.set_description(" Epoch: [{}/{}] | loss: {:.5f}".format(epoch, config['NUM_EPOCH'] - 1, loss.data.item() * batch_multiplier))
		progressbar.update(1)

	lr_scheduler.step()
	progressbar.close()

	epoch_loss = running_loss / float(len_data)
	print(' Epoch over Loss: {:.5f}'.format(epoch_loss))
	writer.add_scalars("train/loss", {branch: epoch_loss}, epoch)
	writer.add_scalars("train/learning_rate", {branch: optimizer.param_groups[0]['lr']}, epoch)

	# SAVE MODEL
	save_model(exp_dir_num, epoch, epoch_loss, model, optimizer, lr_scheduler, branch)

@torch.no_grad()
def val_one_epoch(epoch, branch, model, data_loader, test_model = None):

	print(" Validating {} model".format(branch, epoch))

	model.eval()
	
	gt = torch.FloatTensor()
	pred = torch.FloatTensor()

	running_loss = 0.
	len_data = len(data_loader)
	random_int = int(torch.randint(0, len_data, (1,))[0])
	print(" display images on index", random_int)

	progressbar = tqdm(range(len_data))
	for i, (images, targets) in enumerate(data_loader):

		if i == random_int:
			images_draw = {}
			images_draw['images'] = images.data
			images_draw['targets'] = targets.data

		if branch == 'local':
			output_global = test_model(images.to(device))
			output_patches = AttentionGenPatchs(images.data, output_global['features'].data.cpu(), exp_config['threshold'], exp_config['L_function'])
			images = output_patches['crop']

			del output_global
			torch.cuda.empty_cache()
		
		elif branch == 'fusion':
			output_global = test_model[0](images.to(device))
			output_patches = AttentionGenPatchs(images.data, output_global['features'].data.cpu(), exp_config['threshold'], exp_config['L_function'])
			output_local = test_model[1](output_patches['crop'].to(device))
			images = torch.cat((output_global['pool'], output_local['pool']), dim = 1)

			del output_global, output_local
			torch.cuda.empty_cache()

		images = images.to(device)
		targets = targets.to(device)
		gt = torch.cat((gt, targets.data.cpu()), 0)

		output = model(images)
		pred = torch.cat((pred, torch.sigmoid(output['out']).data.cpu()), 0)

		if i == random_int:
			if branch == 'global':
				draw_image = drawImage(images_draw['images'], images_draw['targets'], torch.sigmoid(output['out']).data)
			else:
				draw_image = drawImage(images_draw['images'],
										images_draw['targets'],
										torch.sigmoid(output['out']).data,
										output_patches['crop'].data,
										output_patches['heatmap'].data,
										output_patches['coordinate'])

			writer.add_images("val/{}".format(branch), draw_image, epoch)

		progressbar.set_description(" Epoch: [{}/{}]".format(epoch,  config['NUM_EPOCH'] - 1))
		progressbar.update(1)

	progressbar.close()

	AUROCs = compute_AUCs(gt, pred)
	AUROCs_mean = np.array(AUROCs).mean()

	writer.add_scalars("val/AUROCs", {branch: AUROCs_mean}, epoch)

	if AUROCs_mean > BEST_AUROCs[branch]:
		BEST_AUROCs[branch] = AUROCs_mean
		save_name = path.join(exp_dir_num, args.exp_num + '_' + branch + '.pth')
		copy_name = os.path.join(exp_dir_num, args.exp_num + '_' + branch + '_best.pth')
		shutil.copyfile(save_name, copy_name)
		print(" Best model is saved: {}".format(copy_name))

	print(' Best AUROCs: {:.5f}'.format(BEST_AUROCs[branch]))
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

	# ================= MODELS ================= #
	print("\n Model initialization")
	print(" ============================================")
	model_param = exp_config['net']
	if 'global' in exp_config['branch']:
		print(" Global branch")
		GlobalModel = ResAttCheXNet(pretrained = True, num_classes = 15, **exp_config['net'])
	if 'local' in exp_config['branch']:
		print(" Global branch")
		GlobalModel = ResAttCheXNet(pretrained = True, num_classes = 15, **exp_configs[model_param]['net'])
		print(" Local branch")
		LocalModel = ResAttCheXNet(pretrained = True, num_classes = 15, **exp_configs[model_param]['net'])
		print(" L distance function \t:", exp_config['L_function'])
		print(" Threshold \t\t:", exp_config['threshold'])

	if 'fusion' in exp_config['branch']:
		FusionModel = FusionNet(backbone = exp_configs[model_param]['net']['backbone'], num_classes = 15)
		

	# ================= CREATE WEIGHT ================= #
	# count_train_labels = torch.tensor(train_dataset.labels, dtype=torch.float32).sum(axis=0)
	# weight_train = count_train_labels.max() / count_train_labels

	if exp_config['loss'] == 'BCELoss':
		criterion = nn.BCELoss()
	elif exp_config['loss'] == 'WeightedBCELoss':
		criterion = WeightedBCELoss(PosNegWeightIsDynamic = True)
	else:
		raise Exception("loss function must be BCELoss or WeightedBCELoss")

	print(" Loss function \t\t:", exp_config['loss'])
	print()

	for branch_name in BRANCH_NAMES:
		start_time_train = time.time()
		# ================= LOAD DATASET ================= #
		train_dataset = ChestXrayDataSet(data_dir = data_dir, split = 'train', num_classes = 15, transform = transform_train)
		train_loader = DataLoader(dataset = train_dataset, batch_size = MAX_BATCH_CAPACITY[branch_name], shuffle = True, num_workers = 4, pin_memory = True)

		val_dataset = ChestXrayDataSet(data_dir = data_dir, split = 'test', num_classes = 15, transform = transform_test)
		val_loader = DataLoader(dataset = val_dataset, batch_size = MAX_BATCH_CAPACITY[branch_name], shuffle = False, num_workers = 4, pin_memory = True)

		# test_dataset = ChestXrayDataSet(data_dir = data_dir, split = 'test', num_classes = config['net']['num_classes'], transform = transform_test)
		# test_loader = DataLoader(dataset = test_dataset, batch_size = config['batch_size']['global'] // 2, shuffle = False, num_workers = 4, pin_memory = True)

		print(" Start training " + branch_name + " branch...")
	
		if branch_name == 'global':
			Model = GlobalModel.to(device)
			TestModel = None

		if branch_name == 'local':
			save_dict_global = torch.load(os.path.join(args.exp_dir, model_param, model_param + '_global_best' + '.pth'))
			GlobalModel.load_state_dict(save_dict_global['net'])

			Model = LocalModel.to(device)
			TestModel = GlobalModel.to(device)

		if branch_name == 'fusion':
			save_dict_global = torch.load(os.path.join(args.exp_dir, model_param, model_param + '_global_best' + '.pth'))
			save_dict_local = torch.load(os.path.join(exp_dir_num, args.exp_num + '_local_best' + '.pth'))

			GlobalModel.load_state_dict(save_dict_global['net'])
			LocalModel.load_state_dict(save_dict_local['net'])

			Model = FusionModel.to(device)
			TestModel = (GlobalModel.to(device), LocalModel.to(device))

		optimizer = optim.SGD(Model.parameters(), **config['optimizer']['SGD'])
		lr_scheduler = optim.lr_scheduler.StepLR(optimizer , **config['lr_scheduler'])

		if args.resume:

			checkpoint = path.join(exp_dir_num, args.exp_num + '_' + branch_name + '.pth')

			if path.isfile(checkpoint):
				save_dict = torch.load(checkpoint)
				Model.load_state_dict(save_dict['net'])
				optimizer.load_state_dict(save_dict['optim'])
				lr_scheduler.load_state_dict(save_dict['lr_scheduler'])
				start_epoch = save_dict['epoch']
				print(" Loaded " + branch_name + " branch model checkpoint from epoch " + str(start_epoch))
				start_epoch += 1
			else:
				start_epoch = 0

		else:
			start_epoch = 0

		for epoch in range(start_epoch, config['NUM_EPOCH']):
			start_time_epoch = time.time()

			train_one_epoch(epoch, branch_name, Model, optimizer, lr_scheduler, train_loader, criterion, TestModel)
			val_one_epoch(epoch, branch_name, Model, val_loader, TestModel)

			end_time_epoch = time.time()
			total_menit = (end_time_epoch - start_time_epoch) / 60
			print(" Training epoch: {:.4f} jam ({:.2f} menit)".format(total_menit / 60, total_menit))

		print(" Training " + branch_name + " branch done")

		end_time_train = time.time()
		total_menit = (end_time_train - start_time_train) / 60
		print(" Training time {} branch: {:.4f} jam ({:.2f} menit)".format(branch_name, total_menit / 60, total_menit))

if __name__ == "__main__":
	main()
