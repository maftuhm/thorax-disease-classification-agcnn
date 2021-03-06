import os
import cv2
import csv
import shutil
import numpy as np
from PIL import Image, ImageDraw

from scipy import signal
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from skimage.measure import label
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms_th
from config import *
from utils import transforms as transforms_
from utils.attention_mask_inference import AttentionMaskInference

def save_model(exp_dir, epoch, auroc, loss, model, optimizer, lr_scheduler, branch_name = 'global'):
	save_dict = {
		"epoch": epoch,
		"auroc": auroc,
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

def compute_ROCs(gt, pred):
	gt_np = gt.cpu().numpy()
	pred_np = pred.cpu().numpy()

	results = {
		'fpr' : [],
		'tpr' : [],
		'thresholds': [],
		'gmeans': [],
		'idmax': [],
		'optimal_threshold': []
	}

	for i in range(len(gt_np[0])):
		fpr, tpr, thresholds = roc_curve(gt_np[:, i], pred_np[:, i])
		gmeans = np.sqrt(tpr * (1-fpr))
		ix = np.argmax(gmeans)
		results['fpr'].append(fpr.tolist())
		results['tpr'].append(tpr.tolist())
		results['thresholds'].append(thresholds.tolist())
		results['gmeans'].append(gmeans.tolist())
		results['idmax'].append(int(ix))
		results['optimal_threshold'].append(float(thresholds[ix]))

	return results

def get_threshold(gt, pred):
	gt_np = gt.cpu().numpy()
	pred_np = pred.cpu().numpy()

	results = []
	for i in range(len(gt_np[0])):
		fpr, tpr, thresholds = roc_curve(gt_np[:, i], pred_np[:, i])
		gmeans = np.sqrt(tpr * (1-fpr))
		ix = np.argmax(gmeans)
		results.append(thresholds[ix])

	return results

def create_roc_curve(gt, pred):
	gt_np = gt.cpu().numpy()
	pred_np = pred.cpu().numpy()

	colors = [
	'#0033cc', '#ff0000', '#ff9933', '#993399', '#009933', '#6699ff', '#cc3300',
	'#006600', '#660066', '#00ffcc', '#ffff00', '#ff33cc', '#00cc99', '#660033']

	linestyles = [
	'solid', 'dotted', 'dashed', 'dashdot',
	(0, (1, 1)), (0, (1, 5)), (0, (5, 5)), (0, (5, 1)),
	(0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)), 'solid', 'dashdot',
	(0, (3, 5, 1, 5, 1, 5)), (0, (3, 1, 1, 1, 1, 1))]

	for i in range(len(gt_np[0])):
		fpr, tpr, thresholds = roc_curve(gt_np[:, i], pred_np[:, i])
		gmeans = np.sqrt(tpr * (1-fpr))
		ix = np.argmax(gmeans)
		plt.plot([0,1], [0,1], linestyle='--')
		plt.plot(fpr, tpr, linestyle=linestyles[i], color=colors[i], label='{} t={:.5f}'.format(CLASS_NAMES[i], thresholds[ix]))
		plt.scatter(fpr[ix], tpr[ix], marker='o', color=colors[i])

	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend()
	plt.show()

def create_precision_recall_curve(gt, pred):
	gt_np = gt.cpu().numpy()
	pred_np = pred.cpu().numpy()

	colors = [
	'#0033cc', '#ff0000', '#ff9933', '#993399', '#009933', '#6699ff', '#cc3300',
	'#006600', '#660066', '#00ffcc', '#ffff00', '#ff33cc', '#00cc99', '#660033']

	linestyles = [
	'solid', 'dotted', 'dashed', 'dashdot',
	(0, (1, 1)), (0, (1, 5)), (0, (5, 5)), (0, (5, 1)),
	(0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)), 'solid', 'dashdot',
	(0, (3, 5, 1, 5, 1, 5)), (0, (3, 1, 1, 1, 1, 1))]

	for i in range(len(gt_np[0])):
		precision, recall, thresholds = precision_recall_curve(gt_np[:, i], pred_np[:, i])
		fscore = (2 * precision * recall) / (precision + recall)
		ix = np.argmax(fscore)
		plt.plot([0,1], [0,1], linestyle='--')
		plt.plot(recall, precision, linestyle=linestyles[i], color=colors[i], label='{} t={:.5f} F={:.3f}'.format(CLASS_NAMES[i], thresholds[ix], fscore[ix]))
		plt.scatter(recall[ix], precision[ix], marker='o', color=colors[i])

	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend()
	plt.show()

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

def draw_bounding_box(image, bbox, label = None, color=(0, 255, 0)):
	img_to_draw = transforms_th.ToPILImage()(image).convert('RGB')
	draw = ImageDraw.Draw(img_to_draw)
	if isinstance(bbox[0], list):
		for bb in bbox:
			draw.rectangle(bb, outline=color)
	else:
		draw.rectangle(bbox, outline=color)

	if label is not None:
		draw.text((bbox[0] + 2, bbox[1]), label)

	return transforms_th.ToTensor()(img_to_draw).unsqueeze(0)

def draw_heatmap(image, heatmap):
	heatmap = np.uint8(255 * heatmap)
	heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

	img = cv2.cvtColor(np.uint8(255 * image.permute(1, 2, 0).numpy()), cv2.COLOR_RGB2BGR)
	add_heatmap = cv2.cvtColor(cv2.addWeighted(img, 0.5, heatmap, 0.5, 0), cv2.COLOR_BGR2RGB)

	return transforms_th.ToTensor()(Image.fromarray(add_heatmap)).unsqueeze(0)

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
	
	return transforms_th.ToTensor()(new_images).unsqueeze(0)

def drawImage(images, target, scores, images_cropped = None, heatmaps = None, gt_coordinates=None, coordinates = None):
	bz, c, h, w = images.shape # batch_size, channel, height, width

	# unnormalize = UnNormalize(mean=[0.4979839647692935], std=[0.22962109349599796]) # ori
	unnormalize = UnNormalize(mean=[0.49886124425113754], std=[0.22925289787072856]) # resize

	img = torch.empty(0, dtype = images.dtype)
	img_scores = torch.empty(0, dtype = images.dtype)

	if images_cropped is not None:
		img_heatmap = torch.empty(0, dtype = images.dtype)
		img_crop = torch.empty(0, dtype = images.dtype)

	if bz > 4: bz = 4

	for i in range(bz):
		img_scores = torch.cat((img_scores, draw_label_score(target[i], scores[i], size = (h, w))), 0)

		if images_cropped is not None:
			image_drawed_bbox = unnormalize(images[i])

			if gt_coordinates is not None:
				if len(gt_coordinates) > 0:
					image_drawed_bbox = draw_bounding_box(image_drawed_bbox, gt_coordinates[i], color=(255, 0, 0))

			img = torch.cat((img, image_drawed_bbox), 0)
			image_drawed_bbox =  draw_bounding_box(draw_heatmap(image_drawed_bbox, heatmaps[i]).squeeze(), coordinates[i])
			img_heatmap = torch.cat((img_heatmap, image_drawed_bbox), 0)
			img_crop = torch.cat((img_crop, unnormalize(images_cropped[i]).unsqueeze(0)), 0)
		else:
			img = torch.cat((img, unnormalize(images[i]).unsqueeze(0)), 0)

	if img.dim() == 3:
		img = img.unsqueeze(1)

	if img.shape[1] == 1:
		img = img.repeat(1, 3, 1, 1)

	if images_cropped is not None:
		if img_crop.shape[1] == 1:
			img_crop = img_crop.repeat(1, 3, 1, 1)

		new_img = torch.cat((img, img_heatmap, img_crop, img_scores), 0)
	else:
		new_img = torch.cat((img, img_scores), 0)

	return torchvision.utils.make_grid(new_img, nrow = bz).unsqueeze(0)

def write_csv(filename, data, mode = 'a'):
	with open(filename, mode = mode, newline = '') as file: 
		writer = csv.writer(file)
		writer.writerow(data)

def reduce_weight_bias(weight, bias, num_classes = 14):
    weight_np = weight.numpy()
    bias_np = bias.numpy()
    
    pca = PCA(n_components=num_classes)
    pca.fit(weight_np)
    weight_np = pca.components_
    bias_np = signal.resample(bias_np, num_classes)
    return torch.from_numpy(weight_np), torch.from_numpy(bias_np)

def get_weight_wbce_loss(labels):
	count_labels = torch.FloatTensor(labels).sum(axis=0)
	return (len(labels) / count_labels) - 1

class Predict:
	def __init__(self, model, transform = None, threshold = None, num_classes = 14, att_mask = None):
		assert len(threshold) == num_classes, "len threshold must be the same with len num classes"
		self.threshold = threshold
		self.model = model
		self.model.eval()
		self.model.requires_grad_(False)

		if att_mask is not None:
			assert isinstance(att_mask, AttentionMaskInference)
			self.attention_mask = att_mask
		else:
			self.attention_mask = AttentionMaskInference(threshold = 0.7, distance_function = 'L2')
		
		if transform is not None:
			self.transform = transform
		else:
			self.transform = transforms_.Compose(
				transforms_.Resize((256, 256)),
				transforms_.CenterCrop((224, 224)),
				transforms_.DynamicNormalize()
			)

		self.results = None

	def predict(self, x):
		x = self.transform(x)
		x = x.to(self.model.device)
		self.results = self.model(x)
		return self.results['score']

	def predict_classes_(self, x):
		assert x is not None, "input x must be not None"
		assert self.threshold is not None, "threshold must be not None."

		x = self.transform(x)
		scores = self.predict(x)
		classes_predicted = []
		for score in scores:
			class_predict = [i for i, (sc, th) in enumerate(zip(score, self.threshold)) if sc > th]
			classes_predicted.append(class_predict)

		return classes_predicted

	def predict_classes(self, x = None, threshold = None):

		if threshold is not None:
			self.threshold = threshold

		assert self.threshold is not None, "threshold must be not None."

		if x is not None:
			x = self.transform(x)
			scores = self.predict(x)
		else:
			assert self.results is not None, "input x must be not None and model.predict(x) never be called."
			scores = self.results['score']

		classes_predicted = []
		for score in scores:
			class_predict = [i for i, (sc, th) in enumerate(zip(score, self.threshold)) if sc > th]
			classes_predicted.append(class_predict)

		return classes_predicted

	def attention(self, x = None):

		if x is not None:
			x = self.transform(x)
			self.predict(x)

		assert self.results is not None, "input x must be not None and model.predict(x) never be called."
		return self.attention_mask(x, self.results['features'])
	
	def features(self, x = None):
		# cooming soon
		pass

	def __call__(self, img, target = None, bbox = None):
		# cooming soon
		pass