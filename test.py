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
# from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

# import torchvision.transforms as transforms

from model import MainNet, FusionNet
from utils import WeightedBCELoss, AttentionMaskInference, ChestXrayDataSet
from utils import transforms

from utils.utils import *
from config import *

def parse_args():
	parser = argparse.ArgumentParser(description='AG-CNN')
	parser.add_argument("--exp_dir", type=str, default='./final_experiments', help='define experiment directory (ex: /exp16)')
	parser.add_argument("--exp_num", type=str, default='exp0', help='define experiment directory (ex: /exp0)')
	parser.add_argument("--dataset", type=str, default='test', help='define dataset name (ex: train, val or test)')
	parser.add_argument("--datadir", type=str, default='../lung-disease-detection/data', help='define data directory (ex: /data)')
	parser.add_argument("--best_model", "-b", action="store_true")
	parser.add_argument("--loss", "-l", action="store_true")
	parser.add_argument("--all_branch",  "-all", action="store_true")
	args = parser.parse_args()
	return args

args = parse_args()

# Load config json file
with open(os.path.join(args.exp_dir, "cfg.json")) as f:
	config = json.load(f)

with open(os.path.join(args.exp_dir, "cfg_train.json")) as f:
	exp_configs = json.load(f)
	exp_config = exp_configs[args.exp_num]

if args.all_branch:
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

exp_dir_num = os.path.join(args.exp_dir, args.exp_num)

if 'num_classes' in list(config.keys()):
	CLASS_NAMES = CLASS_NAMES[:config['num_classes']]

NUM_CLASSES = len(CLASS_NAMES)

MAX_BATCH_CAPACITY = 60
# if MAX_BATCH_CAPACITY % 5 == 0:
# 	num_workers = 5
# else:
num_workers = 4

cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# writer = SummaryWriter(args.exp_dir + '/log')

@torch.no_grad()
def main():
	# ================= TRANSFORMS ================= #
	# normalize = transforms.Normalize(
	#    mean=[0.485, 0.456, 0.406],
	#    std=[0.229, 0.224, 0.225]
	# )
	transform_init = transforms.Resize(tuple(config['dataset']['resize']))
	transform_test = transforms.Compose(
	   transforms.CenterCrop(tuple(config['dataset']['crop'])),
	   transforms.ToTensor(),
	   transforms.DynamicNormalize()
	)

	# ================= LOAD DATASET ================= #

	test_dataset = ChestXrayDataSet(args.datadir, args.dataset, num_classes = NUM_CLASSES, transform = transform_test, init_transform=transform_init)
	test_loader = DataLoader(dataset = test_dataset, batch_size = MAX_BATCH_CAPACITY, shuffle = False, num_workers = num_workers, pin_memory = True)

	# ================= MODELS ================= #
	print("\n Model initialization")
	print(" ============================================")
	print(" Global branch")
	GlobalModel = MainNet(pretrained = False, num_classes = NUM_CLASSES, **config['net']).to(device)
	if args.all_branch:
		print(" Local branch")
		LocalModel = MainNet(pretrained = False, num_classes = NUM_CLASSES, **config['net']).to(device)
		AttentionGenPatchs = AttentionMaskInference(threshold = config['threshold'], distance_function = config['L_function'], keepratio=False)
		FusionModel = FusionNet(backbone = config['net']['backbone'], num_classes = NUM_CLASSES).to(device)
		print(" L distance function \t:", config['L_function'])
		print(" Threshold \t\t:", AttentionGenPatchs.threshold)
	print(" Num classes \t\t:", NUM_CLASSES)
	print(" Optimizer \t\t:", list(config['optimizer'].keys())[0])
	# print(" Lr Scheduler \t\t:", list(config['lr_scheduler'].keys())[0])
	print(" Loss function \t\t:", config['loss'])
	print()

	if args.best_model:
		add_text = '_best'

		if args.loss:
			add_text += '_loss'
		else:
			add_text += '_auroc'
	else:
		add_text = ''

	checkpoint_global = os.path.join(exp_dir_num, args.exp_num + '_global' + add_text + '.pth')
	if args.all_branch:
		checkpoint_global = os.path.join(args.exp_dir, global_branch_exp, global_branch_exp + '_global_best_auroc.pth')
		checkpoint_local = os.path.join(exp_dir_num, args.exp_num + '_local_best_auroc.pth')
		checkpoint_fusion = os.path.join(exp_dir_num, args.exp_num + '_fusion' + add_text + '.pth')

	if os.path.isfile(checkpoint_global):
		save_dict = torch.load(checkpoint_global)
		GlobalModel.load_state_dict(save_dict['net'])
		print(" Loaded Global Branch Model checkpoint from epoch", save_dict['epoch'])
		del save_dict
		torch.cuda.empty_cache()
	else:
		raise Exception("Global Model does not exist")

	if args.all_branch:
		if os.path.isfile(checkpoint_local):
			save_dict = torch.load(checkpoint_local)
			LocalModel.load_state_dict(save_dict['net'])
			print(" Loaded Local Branch Model checkpoint from epoch", save_dict['epoch'])
			AttentionGenPatchs.load_weight(GlobalModel.fc.weight.detach().cpu())
			del save_dict
			torch.cuda.empty_cache()
		else:
			raise Exception("Local Model does not exist")

		if os.path.isfile(checkpoint_fusion):
			save_dict = torch.load(checkpoint_fusion)
			FusionModel.load_state_dict(save_dict['net'])
			print(" Loaded Fusion Branch Model checkpoint from epoch", save_dict['epoch'])
			del save_dict
			torch.cuda.empty_cache()
		else:
			raise Exception("Fusion Model does not exist")

	GlobalModel.eval()

	if args.all_branch:
		LocalModel.eval()
		FusionModel.eval()

	ground_truth = torch.FloatTensor()
	pred_global = torch.FloatTensor()
	if args.all_branch:
		pred_local = torch.FloatTensor()
		pred_fusion = torch.FloatTensor()

	progressbar = tqdm(range(len(test_loader)))

	start_time_test = datetime.now()
	for i, (image, target) in enumerate(test_loader):
		# compute output
		output_global = GlobalModel(image.to(device))

		if args.all_branch:
			output_patches = AttentionGenPatchs(image.detach(), output_global['features'].detach().cpu())

			output_local = LocalModel(output_patches['image'].to(device))

			pool = torch.cat((output_global['pool'], output_local['pool']), dim = 1)
			output_fusion = FusionModel(pool.to(device))

		ground_truth = torch.cat((ground_truth, target.detach()), 0)
		pred_global = torch.cat((pred_global.detach(), output_global['score'].detach().cpu()), 0)

		if args.all_branch:
			pred_local = torch.cat((pred_local.detach(), output_local['score'].detach().cpu()), 0)
			pred_fusion = torch.cat((pred_fusion.detach(), output_fusion['score'].detach().cpu()), 0)

		# if (i + 1) % 300 == 0:
		# 	draw_image = drawImage(image, target, output_fusion.detach().cpu(), image_patch.detach(), heatmaps, coordinates)
		# 	writer.add_images("Val/epoch_{}".format(epoch), draw_image, i + 1)

		progressbar.update(1)

	progressbar.close()

	print(" Testing over time {}:".format(datetime.now() - start_time_test))

	AUROCs_global = compute_AUCs(ground_truth, pred_global)
	AUROCs_global_avg = np.array(AUROCs_global[:14]).mean()

	ROCs_global = compute_ROCs(ground_truth, pred_global)

	AUROCs_local = [0. for a in range(NUM_CLASSES)]
	AUROCs_local_avg = 0.
	AUROCs_fusion = [0. for a in range(NUM_CLASSES)]
	AUROCs_fusion_avg = 0.

	if args.all_branch:
		AUROCs_local = compute_AUCs(ground_truth, pred_local)
		AUROCs_local_avg = np.array(AUROCs_local[:14]).mean()
		ROCs_local = compute_ROCs(ground_truth, pred_local)
		AUROCs_fusion = compute_AUCs(ground_truth, pred_fusion)
		AUROCs_fusion_avg = np.array(AUROCs_fusion[:14]).mean()
		ROCs_fusion = compute_ROCs(ground_truth, pred_fusion)

	write_csv(os.path.join(exp_dir_num, args.exp_num + '_AUROCs_' + args.dataset + add_text + '.csv'),
						data = ['Model'] + CLASS_NAMES + ['Mean'], mode = 'w')

	write_csv(os.path.join(exp_dir_num, args.exp_num + '_AUROCs_' + args.dataset + add_text + '.csv'),
						data = ['Global'] + list(AUROCs_global) + [AUROCs_global_avg], mode = 'a')

	with open(os.path.join(exp_dir_num, args.exp_num + '_ROCs_global_' + args.dataset + add_text + '.json'), 'w') as f:
		json.dump(ROCs_global, f)

	if args.all_branch:
		write_csv(os.path.join(exp_dir_num, args.exp_num + '_AUROCs_' + args.dataset + add_text + '.csv'),
							data = ['Local'] + list(AUROCs_local) + [AUROCs_local_avg],
							mode = 'a')
		with open(os.path.join(exp_dir_num, args.exp_num + '_ROCs_local_' + args.dataset + add_text + '.json'), 'w') as f:
			json.dump(ROCs_local, f)

		write_csv(os.path.join(exp_dir_num, args.exp_num + '_AUROCs_' + args.dataset + add_text + '.csv'),
							data = ['Fusion'] + list(AUROCs_fusion) + [AUROCs_fusion_avg],
							mode = 'a')

		with open(os.path.join(exp_dir_num, args.exp_num + '_ROCs_fusion_' + args.dataset + add_text + '.json'), 'w') as f:
			json.dump(ROCs_fusion, f)

	print("|===============================================================================================|")
	print("|\t\t\t\t\t    AUROC\t\t\t\t\t\t|")
	print("|===============================================================================================|")
	print("|\t\t\t|  Global branch\t|  Local branch\t\t|  Fusion branch\t|")
	print("|-----------------------------------------------------------------------------------------------|")

	for i in range(NUM_CLASSES):
		if len(CLASS_NAMES[i]) < 6:
			print("| {}\t\t\t|".format(CLASS_NAMES[i]), end="")
		elif len(CLASS_NAMES[i]) > 14:
			print("| {}\t|".format(CLASS_NAMES[i]), end="")
		else:
			print("| {}\t\t|".format(CLASS_NAMES[i]), end="")
		print("  {:.10f}\t\t|  {:.10f}\t\t|  {:.10f}\t\t|".format(AUROCs_global[i], AUROCs_local[i], AUROCs_fusion[i]))
	print("|-----------------------------------------------------------------------------------------------|")
	print("| Average\t\t|  {:.10f}\t\t|  {:.10f}\t\t|  {:.10f}\t\t|".format(AUROCs_global_avg, AUROCs_local_avg, AUROCs_fusion_avg))
	print("|===============================================================================================|")
	print()
	# create_precision_recall_curve(ground_truth, pred_global)

if __name__ == "__main__":
	main()
