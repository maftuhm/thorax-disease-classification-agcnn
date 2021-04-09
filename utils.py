import os
import cv2
import shutil
import numpy as np
from PIL import Image, ImageDraw

from sklearn.metrics import roc_auc_score
from skimage.measure import label

import torch
import torchvision.transforms as transforms

def AttentionGenPatchs(ori_image, fm_cuda, size_crop = (224, 224)):

	feature_conv = fm_cuda.data.cpu().numpy()

	bz, nc, h, w = feature_conv.shape

	images = torch.randn(bz, 3, size_crop[0], size_crop[1])

	for i in range(bz):
		feature = np.abs(feature_conv[i])
		heatmap = np.max(feature, axis = 0)
		# heatmap = (heatmap - np.min(heatmap)) / np.max(heatmap)
		heatmap = np.uint8(255 * heatmap)

		heatmap = cv2.resize(heatmap, size_crop)
		# heatmap[heatmap > 0.7] = 1
		# heatmap[heatmap != 1] = 0
		# _, heatmap_bin = cv2.threshold(resized_heatmap , 0 , 255 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		_, heatmap_bin = cv2.threshold(heatmap , int(255 * 0.7) , 255 , cv2.THRESH_BINARY)
		heatmap_maxconn = selectMaxConnect(heatmap_bin)
		heatmap_mask = heatmap_bin * heatmap_maxconn

		ind = np.argwhere(heatmap_mask != 0)
		minh = min(ind[:,0])
		minw = min(ind[:,1])
		maxh = max(ind[:,0])
		maxw = max(ind[:,1])

		image = ori_image[i].numpy().transpose(1, 2, 0)
		image_crop = image[minh:maxh, minw:maxw, :]
		image_crop = cv2.resize(image_crop, size_crop)
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

def save_model(exp_dir, epoch, model, optimizer, lr_scheduler, branch_name = 'global'):
	save_dict = {
		"epoch": epoch,
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

def drawImage(images, labels, images_cropped, coordinates):
	bz, c, h, w = images.shape # batch_size, channel, height, width

	new_images = Image.new('RGB', (bz * w, 2 * h), (0, 0, 0))

	for i in range(bz):

		img = transforms.ToPILImage()(images[i])
		img_patch = transforms.ToPILImage()(images_cropped[i])

		draw_img = ImageDraw.Draw(img)
		draw_img.rectangle(coordinates, outline=(0, 255, 0))
		draw_img.text((coordinates[0], coordinates[1] - 10), labels[i], (0, 255, 0))

		new_images.paste(img, (i * w, 0))
		new_images.paste(img_patch, (i * w, w))

	new_images = transforms.ToTensor()(new_images).unsqueeze(0)
	return new_images