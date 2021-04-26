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
	parser.add_argument("--exp_dir", type=str, default="./experiments/exp9")
	parser.add_argument("--best_model", "-b", action="store_true")
	args = parser.parse_args()
	return args

args = parse_args()

# Load config json file
with open(path.join(args.exp_dir, "cfg.json")) as f:
	exp_cfg = json.load(f)

# ================= CONSTANTS ================= #
data_dir = path.join('D:/', 'Data', 'data')
classes_name = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
				'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

max_batch_capacity = 4

cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# writer = SummaryWriter(args.exp_dir + '/log')

def main():
	# ================= TRANSFORMS ================= #
	normalize = transforms.Normalize(
	   mean=[0.485, 0.456, 0.406],
	   std=[0.229, 0.224, 0.225]
	)

	transform_test = transforms.Compose([
	   transforms.Resize(tuple(exp_cfg['dataset']['resize'])),
	   transforms.CenterCrop(tuple(exp_cfg['dataset']['crop'])),
	   transforms.ToTensor(),
	   normalize,
	])

	# ================= LOAD DATASET ================= #
	val_dataset = ChestXrayDataSet(data_dir = data_dir, split = 'val', transform = transform_test)
	val_loader = DataLoader(dataset = val_dataset, batch_size = 32, shuffle = False, num_workers = 4, pin_memory = True)

	test_dataset = ChestXrayDataSet(data_dir = data_dir, split = 'test', transform = transform_test)
	test_loader = DataLoader(dataset = test_dataset, batch_size = 32, shuffle = False, num_workers = 4, pin_memory = True)

	# ================= MODELS ================= #
	GlobalModel = Net(exp_cfg['backbone']).to(device)
	LocalModel = Net(exp_cfg['backbone']).to(device)
	FusionModel = FusionNet(exp_cfg['backbone']).to(device)

	if args.best_model:
		add_text = '_best'
	else:
		add_text = ''

	checkpoint_global = path.join(args.exp_dir, args.exp_dir.split('/')[-1] + '_global' + add_text + '.pth')
	checkpoint_local = path.join(args.exp_dir, args.exp_dir.split('/')[-1] + '_local' + add_text + '.pth')
	checkpoint_fusion = path.join(args.exp_dir, args.exp_dir.split('/')[-1] + '_fusion' + add_text + '.pth')

	if path.isfile(checkpoint_global):
		save_dict = torch.load(checkpoint_global)
		GlobalModel.load_state_dict(save_dict['net'])
		print(" Loaded Global Branch Model checkpoint from epoch", save_dict['epoch'])

	if path.isfile(checkpoint_local):
		save_dict = torch.load(checkpoint_local)
		LocalModel.load_state_dict(save_dict['net'])
		print(" Loaded Local Branch Model checkpoint from epoch", save_dict['epoch'])

	if path.isfile(checkpoint_fusion):
		save_dict = torch.load(checkpoint_fusion)
		FusionModel.load_state_dict(save_dict['net'])
		print(" Loaded Fusion Branch Model checkpoint from epoch", save_dict['epoch'])

	write_csv(path.join(args.exp_dir, args.exp_dir.split('/')[-1] + '_AUROCs.csv'),
						data = ['Model'] + classes_name + ['Mean'],
						mode = 'w')

	GlobalModel.eval()
	LocalModel.eval()
	FusionModel.eval()

	ground_truth = torch.FloatTensor()
	pred_global = torch.FloatTensor()
	pred_local = torch.FloatTensor()
	pred_fusion = torch.FloatTensor()

	progressbar = tqdm(range(len(test_loader)))

	with torch.no_grad():
		for i, (image, target) in enumerate(test_loader):

			# compute output
			output_global, fm_global, pool_global = GlobalModel(image.to(device))
			
			image_patch, heatmaps, coordinates = AttentionGenPatchs(image.detach(), fm_global.detach().cpu())

			output_local, _, pool_local = LocalModel(image_patch.to(device))

			output_fusion = FusionModel(pool_global, pool_local)

			ground_truth = torch.cat((ground_truth, target.detach()), 0)
			pred_global = torch.cat((pred_global.detach(), output_global.detach().cpu()), 0)
			pred_local = torch.cat((pred_local.detach(), output_local.detach().cpu()), 0)
			pred_fusion = torch.cat((pred_fusion.detach(), output_fusion.detach().cpu()), 0)

			# if (i + 1) % 300 == 0:
			# 	draw_image = drawImage(image, target, output_fusion.detach().cpu(), image_patch.detach(), heatmaps, coordinates)
			# 	writer.add_images("Val/epoch_{}".format(epoch), draw_image, i + 1)

			progressbar.update(1)

		progressbar.close()

	AUROCs_global = compute_AUCs(ground_truth, pred_global)
	AUROCs_global_avg = np.array(AUROCs_global).mean()
	AUROCs_local = compute_AUCs(ground_truth, pred_local)
	AUROCs_local_avg = np.array(AUROCs_local).mean()
	AUROCs_fusion = compute_AUCs(ground_truth, pred_fusion)
	AUROCs_fusion_avg = np.array(AUROCs_fusion).mean()

	write_csv(path.join(args.exp_dir, args.exp_dir.split('/')[-1] + '_AUROCs.csv'),
						data = ['Global'] + list(AUROCs_global) + [AUROCs_global_avg],
						mode = 'a')
	write_csv(path.join(args.exp_dir, args.exp_dir.split('/')[-1] + '_AUROCs.csv'),
						data = ['Local'] + list(AUROCs_local) + [AUROCs_local_avg],
						mode = 'a')
	write_csv(path.join(args.exp_dir, args.exp_dir.split('/')[-1] + '_AUROCs.csv'),
						data = ['Fusion'] + list(AUROCs_fusion) + [AUROCs_fusion_avg],
						mode = 'a')

	print("|===============================================================================================|")
	print("|\t\t\t\t\t    AUROC\t\t\t\t\t\t|")
	print("|===============================================================================================|")
	print("|\t\t\t|  Global branch\t|  Local branch\t\t|  Fusion branch\t|")
	print("|-----------------------------------------------------------------------------------------------|")

	for i in range(len(classes_name)):
		if len(classes_name[i]) < 6:
			print("| {}\t\t\t|".format(classes_name[i]), end="")
		elif len(classes_name[i]) > 14:
			print("| {}\t|".format(classes_name[i]), end="")
		else:
			print("| {}\t\t|".format(classes_name[i]), end="")
		print("  {:.10f}\t\t|  {:.10f}\t\t|  {:.10f}\t\t|".format(AUROCs_global[i], AUROCs_local[i], AUROCs_fusion[i]))
	print("|-----------------------------------------------------------------------------------------------|")
	print("| Average\t\t|  {:.10f}\t\t|  {:.10f}\t\t|  {:.10f}\t\t|".format(AUROCs_global_avg, AUROCs_local_avg, AUROCs_fusion_avg))
	print("|===============================================================================================|")
	print()

if __name__ == "__main__":
	main()