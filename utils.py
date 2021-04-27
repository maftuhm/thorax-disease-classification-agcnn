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
import torchvision.transforms as transforms

CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
				'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

def L1(feature):
	output = torch.abs(feature)
	output = torch.sum(output, axis = 0) / feature.shape[0]
	return output

def L2(feature):
	output = torch.sum(feature ** 2, axis = 0)
	output = torch.sqrt(output) / feature.shape[0]
	return output

def Lmax(feature):
	output = torch.abs(feature)
	output = torch.max(output, dim = 0)[0]
	return output

def Lmax_normalize(feature):
	output = Lmax(feature)
	max1 = torch.max(output)
	min1 = torch.min(output)
	output = (output - min1) / (max1 - min1)
	return output

def L3(feature):
	output = torch.sum(feature, axis = 0)
	output = output - torch.min(output)
	output = output / torch.max(output)
	return output

def AttentionGenPatchs(ori_image, features_global, threshold = 0.7):

	batch_size = features_global.shape[0]
	n, c, h, w = ori_image.shape

	cropped_image = torch.zeros(batch_size, c, w, h, dtype=ori_image.dtype)
	heatmaps = torch.zeros(batch_size, c, w, h, dtype=ori_image.dtype)
	coordinates = []

	for b in range(batch_size):
		heatmap = L3(features_global[b])

		heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), size=(h, w), mode = 'bilinear', align_corners = True)
		heatmap = torch.squeeze(heatmap)
		heatmaps[b] = heatmap
		heatmap[heatmap > threshold] = 1
		heatmap[heatmap != 1] = 0

		heatmap = selectMaxConnect(heatmap)

		where = torch.from_numpy(np.argwhere(heatmap == 1))
		if len(where) < 1:
			xmin, xmax, ymin, ymax = 0, w, 0, h
			cropped_image[b] = ori_image[b]
		else:
			xmin = int(torch.min(where, dim = 0)[0][0])
			xmax = int(torch.max(where, dim = 0)[0][0])
			ymin = int(torch.min(where, dim = 0)[0][1])
			ymax = int(torch.max(where, dim = 0)[0][1])

			img_crop = ori_image[b][:, xmin:xmax, ymin:ymax]
			cropped_image[b] = F.interpolate(img_crop.unsqueeze(0), size=(h, w), mode = 'bilinear', align_corners = True).squeeze(0)

		coordinates.append((xmin, ymin, xmax, ymax))

	output = {
		'crop' : cropped_image,
		'heatmap' : heatmaps,
		'coordinate' : coordinates
	}

	return output


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

def save_model(exp_dir, epoch, loss, model, optimizer, lr_scheduler, branch_name = 'global'):
	save_dict = {
		"epoch": epoch,
		"loss": loss,
		"net": model.state_dict(),
		"optim": optimizer.state_dict(),
		"lr_scheduler": lr_scheduler.state_dict()
	}
	save_name = os.path.join(exp_dir, exp_dir.split('/')[-1] + '_' + branch_name + '.pth')
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
	for i in range(14):
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
		for t, m, s in zip(tensor, self.mean, self.std):
			t.mul_(s).add_(m)
			# The normalize code -> t.sub_(m).div_(s)
		return tensor

def drawImage(images, target, scores, images_cropped, heatmaps, coordinates):
	bz, c, h, w = images.shape # batch_size, channel, height, width

	new_images = Image.new('RGB', (bz * w, 4 * h), (255, 255, 255))
	
	unnormalize = UnNormalize(
		mean=[0.485, 0.456, 0.406],
		std=[0.229, 0.224, 0.225]
	)

	for i in range(bz):
		img = unnormalize(images[i])
		img = transforms.ToPILImage()(img)

		heatmap = transforms.ToPILImage()(heatmaps[i])

		img_patch = unnormalize(images_cropped[i])
		img_patch = transforms.ToPILImage()(img_patch)

		draw_img = ImageDraw.Draw(img)
		draw_img.rectangle(coordinates[i], outline=(0, 255, 0))
		# draw_img.text((coordinates[i][0] + 5, coordinates[i][1] + 5), str(labels[i]), (0, 255, 0))

		new_images.paste(img, (i * w, 0))
		new_images.paste(heatmap, (i * w, w))
		new_images.paste(img_patch, (i * w, w * 2))

		draw_img = ImageDraw.Draw(new_images)
		draw_img.rectangle((i * w, 2 * h, (i + 1) * w, 4 * h), outline=(0, 0, 0))

		ground_truth_text = [CLASS_NAMES[ind_t] for ind_t, a in enumerate(target[i]) if a != 0]
		predicted_text = [(a, b) for a, b in zip(CLASS_NAMES, scores[i].tolist())]
		predicted_text = sorted(predicted_text, key=lambda x: x[1], reverse=True)

		for j, (text, score) in enumerate(predicted_text):
			if text in ground_truth_text:
				fill = (0, 0, 255)
			else:
				fill = (0, 0, 0)
			draw_img.text((i * w + 15, w * 3 + j * 15 + 10), text, fill = fill)
			draw_img.text((i * w + 150, w * 3 + j * 15 + 10), str(score)[:10], fill = fill)

	new_images = transforms.ToTensor()(new_images).unsqueeze(0)
	return new_images

def write_csv(filename, data, mode = 'a'):
	with open(filename, mode = mode, newline = '') as file: 
		writer = csv.writer(file)
		writer.writerow(data)