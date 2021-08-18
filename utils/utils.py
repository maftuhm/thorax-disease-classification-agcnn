import os
import cv2
import csv
import shutil
import numpy as np
from PIL import Image, ImageDraw

from sklearn.metrics import roc_auc_score
from skimage.measure import label

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from config import *

def save_model(exp_dir, epoch, loss, model, optimizer, lr_scheduler, branch_name = 'global'):
	save_dict = {
		"epoch": epoch,
		"loss": loss,
		"net": model.state_dict(),
		"optim": optimizer.state_dict(),
		"lr_scheduler": lr_scheduler.state_dict()
	}
	save_name = os.path.join(exp_dir, os.path.split(exp_dir)[-1] + '_' + branch_name + '.pth')
	torch.save(save_dict, save_name)
	print(" Model is saved: {}".format(save_name))

def load_model(checkpoint_path, model, optimizer, lr_scheduler):
	if os.path.isfile(checkpoint_path):
		save_dict = torch.load(checkpoint_path)
		epoch = save_dict['epoch']
		loss = save_dict['loss']
		model.load_state_dict(save_dict['net'])
		optimizer.load_state_dict(save_dict['optim'])
		lr_scheduler.load_state_dict(save_dict['lr_scheduler'])
		return epoch, loss, model, optimizer, lr_scheduler
	else:
		return False

def compute_AUCs(gt, pred):
	AUROCs = []
	gt_np = gt.cpu().numpy()
	pred_np = pred.cpu().numpy()
	for i in range(len(gt_np[0])):
		AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
	return AUROCs

class UnNormalize(object):
	def __init__(self, mean, std):
		self.mean = mean
		self.std = std

	def __call__(self, tensor):
		"""
		Args:
			tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
		Returns:
			Tensor: Normalized image.
		"""
		out_tensor = torch.empty(0, dtype = tensor.dtype, device = tensor.device)
		for t, m, s in zip(tensor, self.mean, self.std):
			out_tensor = torch.cat((out_tensor, t.mul(s).add(m).unsqueeze(0)), 0)
			# The normalize code -> t.sub_(m).div_(s)
		return out_tensor

def draw_bounding_box(image, bbox, label = None):
	img_to_draw = transforms.ToPILImage()(image)
	draw = ImageDraw.Draw(img_to_draw)
	draw.rectangle(bbox, outline=(0, 255, 0))

	if label is not None:
		draw.text((bbox[0] + 2, bbox[1]), label)

	return transforms.ToTensor()(img_to_draw).unsqueeze(0)

def draw_heatmap(image, heatmap):
	heatmap = np.uint8(255 * heatmap.numpy())
	heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

	img = cv2.cvtColor(np.uint8(255 * image.permute(1, 2, 0).numpy()), cv2.COLOR_RGB2BGR)
	add_heatmap = cv2.cvtColor(cv2.addWeighted(img, 0.5, heatmap, 0.5, 0), cv2.COLOR_BGR2RGB)

	return transforms.ToTensor()(Image.fromarray(add_heatmap)).unsqueeze(0)

def draw_label_score(target, scores, size = (224, 224)):
	new_images = Image.new('RGB', size, (255, 255, 255))
	draw_img = ImageDraw.Draw(new_images)
	draw_img.rectangle((0, 0, size[0], size[1]), outline=(0, 0, 0))

	ground_truth_text = [CLASS_NAMES[i] for i, val in enumerate(target) if val != 0]
	predicted_text = [(a, b) for a, b in zip(CLASS_NAMES, scores.tolist())]
	predicted_text = sorted(predicted_text, key=lambda x: x[1], reverse=True)

	for i, (text, score) in enumerate(predicted_text):
		if text in ground_truth_text:
			fill = (0, 0, 255)
		else:
			fill = (0, 0, 0)

		draw_img.text((15, i * 15 + 10), text, fill = fill)
		draw_img.text((150, i * 15 + 10), str(score)[:10], fill = fill)
	
	return transforms.ToTensor()(new_images).unsqueeze(0)

def drawImage(images, target, scores, images_cropped = None, heatmaps = None, coordinates = None):
	bz, c, h, w = images.shape # batch_size, channel, height, width

	unnormalize = UnNormalize(
		mean=[0.485, 0.456, 0.406],
		std=[0.229, 0.224, 0.225]
	)
	img = torch.empty(0, dtype = images.dtype)
	img_scores = torch.empty(0, dtype = images.dtype)

	if images_cropped is not None:
		img_heatmap = torch.empty(0, dtype = images.dtype)
		img_crop = torch.empty(0, dtype = images.dtype)

	if bz > 4: bz = 4

	for i in range(bz):
		img_scores = torch.cat((img_scores, draw_label_score(target[i], scores[i], size = (h, w))), 0)

		if images_cropped is not None:
			img = torch.cat((img, draw_bounding_box(unnormalize(images[i]), coordinates[i])), 0)
			img_heatmap = torch.cat((img_heatmap, draw_heatmap(unnormalize(images[i]), heatmaps[i])), 0)
			img_crop = torch.cat((img_crop, unnormalize(images_cropped[i]).unsqueeze(0)), 0)
		else:
			img = torch.cat((img, unnormalize(images[i]).unsqueeze(0)), 0)
	
	if images_cropped is not None:
		new_img = torch.cat((img, img_heatmap, img_crop, img_scores), 0)
	else:
		new_img = torch.cat((img, img_scores), 0)

	return torchvision.utils.make_grid(new_img, nrow = bz).unsqueeze(0)

def write_csv(filename, data, mode = 'a'):
	with open(filename, mode = mode, newline = '') as file: 
		writer = csv.writer(file)
		writer.writerow(data)