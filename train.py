import os
import os.path as path
import json
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

from read_data import ChestXrayDataSet
from model import ResAttCheXNet, FusionNet
from utils import *

def parse_args():
	parser = argparse.ArgumentParser(description='AG-CNN')
	parser.add_argument('--use', type=str, default='train', help='use for what (train or test)')
	parser.add_argument("--exp_dir", type=str, default="./experiments/exp16")
	parser.add_argument("--resume", "-r", action="store_true")
	args = parser.parse_args()
	return args

args = parse_args()

# Load config json file
with open(path.join(args.exp_dir, "cfg.json")) as f:
	exp_cfg = json.load(f)

# ================= CONSTANTS ================= #
data_dir = path.join('..', 'lung-disease-detection', 'data')

BRANCH_NAME_LIST = ['global', 'local', 'fusion']
BEST_VAL_LOSS = {branch: 1000 for branch in BRANCH_NAME_LIST}
# exp 14 best local 0.92933

MAX_BATCH_CAPACITY = {
	'global' : 20,
	'local' : 10,
	'fusion' : 10
}

cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(args.exp_dir + '/log')

def train_one_epoch(epoch, branch, model, optimizer, lr_scheduler, data_loader, test_model = None):

	model.train()
	optimizer.zero_grad()

	running_loss = 0.
	len_data = len(data_loader)
	random_int = int(torch.randint(0, len_data, (1,))[0])
	print(" display images on index", random_int)
	batch_multiplier = exp_cfg['batch_size'][branch] // MAX_BATCH_CAPACITY[branch]

	progressbar = tqdm(range(len_data))
	for i, (images, targets) in enumerate(data_loader):

		if i == random_int:
			images_draw = {}
			images_draw['images'] = images.detach().data
			images_draw['targets'] = targets.detach().data

		if branch == 'local':
			with torch.no_grad():
				output_global = test_model(images.to(device))
				output_patches = AttentionGenPatchs(images.detach(), output_global['features'].cpu())
				images = output_patches['crop']

			del output_global
			torch.cuda.empty_cache()
		
		elif branch == 'fusion':
			with torch.no_grad():
				output_global = test_model[0](images.to(device))
				output_patches = AttentionGenPatchs(images.detach(), output_global['features'].cpu())
				output_local = test_model[1](output_patches['crop'].to(device))
				images = torch.cat((output_global['pool'], output_local['pool']), dim = 1)

			del output_global, output_local
			torch.cuda.empty_cache()

		images = images.to(device)
		targets = targets.to(device)

		output = model(images, targets)

		loss = output['loss'] / batch_multiplier
		running_loss += loss.data.item() * batch_multiplier

		loss.backward()
		if (i + 1) % batch_multiplier == 0:
			optimizer.step()
			optimizer.zero_grad()

		if i == random_int:
			if branch == 'global':
				draw_image = drawImage(images_draw['images'], images_draw['targets'], output['scores'].detach().data)
			else:
				draw_image = drawImage(images_draw['images'],
										images_draw['targets'],
										output['scores'].detach().data,
										output_patches['crop'].detach().data,
										output_patches['heatmap'].detach().data,
										output_patches['coordinate'])

			writer.add_images("train/{}".format(branch), draw_image, epoch)

		progressbar.set_description(" Epoch: [{}/{}] | loss: {:.5f}".format(epoch, exp_cfg['NUM_EPOCH'] - 1, loss.data.item() * batch_multiplier))
		progressbar.update(1)

	lr_scheduler.step()
	progressbar.close()

	epoch_loss = running_loss / float(len_data)
	print(' Epoch over Loss: {:.5f}'.format(epoch_loss))
	writer.add_scalars("train/loss", {branch: epoch_loss}, epoch)
	writer.add_scalars("train/learning_rate", {branch: optimizer.param_groups[0]['lr']}, epoch)

	# SAVE MODEL
	save_model(args.exp_dir, epoch, epoch_loss, model, optimizer, lr_scheduler, branch)

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
			images_draw['images'] = images.detach().data
			images_draw['targets'] = targets.detach().data

		if branch == 'local':
			output_global = test_model(images.to(device))
			output_patches = AttentionGenPatchs(images.detach(), output_global['features'].cpu())
			images = output_patches['crop']

			del output_global
			torch.cuda.empty_cache()
		
		elif branch == 'fusion':
			output_global = test_model[0](images.to(device))
			output_patches = AttentionGenPatchs(images.detach(), output_global['features'].cpu())
			output_local = test_model[1](output_patches['crop'].to(device))
			images = torch.cat((output_global['pool'], output_local['pool']), dim = 1)

			del output_global, output_local
			torch.cuda.empty_cache()

		images = images.to(device)
		targets = targets.to(device)
		gt = torch.cat((gt, targets.detach().cpu()), 0)

		output = model(images, targets)
		pred = torch.cat((pred, output['scores'].detach().cpu()), 0)

		loss = output['loss'].detach().item()

		running_loss += loss

		if i == random_int:
			if branch == 'global':
				draw_image = drawImage(images_draw['images'], images_draw['targets'], output['scores'].detach().data)
			else:
				draw_image = drawImage(images_draw['images'],
										images_draw['targets'],
										output['scores'].detach().data,
										output_patches['crop'].detach().data,
										output_patches['heatmap'].detach().data,
										output_patches['coordinate'])

			writer.add_images("val/{}".format(branch), draw_image, epoch)

		progressbar.set_description(" Epoch: [{}/{}] | loss: {:.5f}".format(epoch,  exp_cfg['NUM_EPOCH'] - 1, loss))
		progressbar.update(1)

	progressbar.close()

	epoch_loss = float(running_loss) / float(len_data)
	writer.add_scalars("val/loss", {branch: epoch_loss}, epoch)
	print(' Epoch over Loss: {:.5f}'.format(epoch_loss))

	if epoch_loss < BEST_VAL_LOSS[branch]:
		BEST_VAL_LOSS[branch] = epoch_loss
		save_name = path.join(args.exp_dir, args.exp_dir.split('/')[-1] + '_' + branch + '.pth')
		copy_name = os.path.join(args.exp_dir, args.exp_dir.split('/')[-1] + '_' + branch + '_best.pth')
		shutil.copyfile(save_name, copy_name)
		print(" Best model is saved: {}".format(copy_name))

	AUROCs = compute_AUCs(gt, pred)
	print(' Best Loss: {:.5f}'.format(BEST_VAL_LOSS[branch]))
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

	# ================= MODELS ================= #
	GlobalModel = ResAttCheXNet(**exp_cfg['net'])
	LocalModel = ResAttCheXNet(**exp_cfg['net'])
	FusionModel = FusionNet(**exp_cfg['net'])

	for branch_name in BRANCH_NAME_LIST:
		print(" Start training " + branch_name + " branch...")

		# ================= LOAD DATASET ================= #
		train_dataset = ChestXrayDataSet(data_dir = data_dir,split = 'train', transform = transform_train)
		train_loader = DataLoader(dataset = train_dataset, batch_size = MAX_BATCH_CAPACITY[branch_name], shuffle = True, num_workers = 4, pin_memory = True)

		val_dataset = ChestXrayDataSet(data_dir = data_dir, split = 'val', transform = transform_test)
		val_loader = DataLoader(dataset = val_dataset, batch_size = exp_cfg['batch_size'][branch_name] // 2, shuffle = False, num_workers = 4, pin_memory = True)

		test_dataset = ChestXrayDataSet(data_dir = data_dir, split = 'test', transform = transform_test)
		test_loader = DataLoader(dataset = test_dataset, batch_size = exp_cfg['batch_size'][branch_name] // 2, shuffle = False, num_workers = 4, pin_memory = True)

		if branch_name == 'global':
			Model = GlobalModel.to(device)
			TestModel = None

		if branch_name == 'local':

			save_dict_global = torch.load(os.path.join(args.exp_dir, args.exp_dir.split('/')[-1] + '_global_best' + '.pth'))
			GlobalModel.load_state_dict(save_dict_global['net'])

			Model = LocalModel.to(device)
			TestModel = GlobalModel.to(device)

		if branch_name == 'fusion':

			save_dict_global = torch.load(os.path.join(args.exp_dir, args.exp_dir.split('/')[-1] + '_global_best' + '.pth'))
			save_dict_local = torch.load(os.path.join(args.exp_dir, args.exp_dir.split('/')[-1] + '_local_best' + '.pth'))

			GlobalModel.load_state_dict(save_dict_global['net'])
			LocalModel.load_state_dict(save_dict_local['net'])

			Model = FusionModel.to(device)
			TestModel = (GlobalModel.to(device), LocalModel.to(device))

		optimizer = optim.SGD(Model.parameters(), **exp_cfg['optimizer']['SGD'])
		lr_scheduler = optim.lr_scheduler.StepLR(optimizer , **exp_cfg['lr_scheduler'])

		if args.resume:

			checkpoint = path.join(args.exp_dir, args.exp_dir.split('/')[-1] + '_' + branch_name + '.pth')

			if path.isfile(checkpoint):
				save_dict = torch.load(checkpoint)
				Model.load_state_dict(save_dict['net'])
				optimizer.load_state_dict(save_dict['optim'])
				lr_scheduler.load_state_dict(save_dict['lr_scheduler'])
				start_epoch = save_dict['epoch']
				BEST_VAL_LOSS[branch_name] = save_dict['loss']
				print(" Loaded " + branch_name + " branch model checkpoint from epoch " + str(start_epoch))
				start_epoch += 1
			else:
				start_epoch = 0

		else:
			start_epoch = 0

		for epoch in range(start_epoch, exp_cfg['NUM_EPOCH']):

			train_one_epoch(epoch, branch_name, Model, optimizer, lr_scheduler, train_loader, TestModel)
			val_one_epoch(epoch, branch_name, Model, val_loader, TestModel)

		print(" Training " + branch_name + " branch done.")

if __name__ == "__main__":
	main()
