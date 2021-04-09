import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

import numpy as np
import skimage.measure as measure

class BackboneNet(nn.Module):
	def __init__(self, model_name, num_classes = 14):
		super(BackboneNet, self).__init__()
		
		if model_name == 'resnet50':
			backbone = models.resnet50(pretrained = True)
			self.features = nn.Sequential(*list(backbone.children())[:-2])
			num_features = backbone.fc.in_features

		if model_name == 'densenet121':
			backbone = models.densenet121(pretrained = True)
			self.features = nn.Sequential(*list(backbone.children())[:-1])
			num_features = backbone.classifier.in_features

		self.maxpool = nn.MaxPool2d(kernel_size = 7, stride = 1)
		self.fc = nn.Sequential(
			nn.Linear(num_features, num_classes),
			nn.Sigmoid()
		)

	def forward(self, x):
		features = self.features(x)
		pool = self.maxpool(features)
		out_after_pooling = pool.view(pool.size(0), -1)
		out = self.fc(out_after_pooling)
		return out, features, out_after_pooling

class FusionNet(nn.Module):
	def __init__(self, model_name, num_classes = 14):
		super(FusionNet, self).__init__()

		if model_name == 'resnet50':
			# num_features = backbone.fc.in_features
			self.fc = nn.Linear(2048 * 2, num_classes)

		if model_name == 'densenet121':
			# num_features = backbone.classifier.in_features
			self.fc = nn.Linear(1028 * 2, num_classes)
		
		self.sigmoid = nn.Sigmoid()

	def forward(self, global_pool, local_pool):
		fusion = torch.cat((global_pool, local_pool), 1)
		out = self.fc(fusion)
		out = self.sigmoid(out)
		return out

class AttentionGuidedNet(nn.Module):
	def __init__(self, model_name, num_classes = 14):
		super(AttentionGuidedNet, self).__init__()

		self.global_net = BackboneNet(model_name, num_classes)
		self.local_net = BackboneNet(model_name, num_classes)
		self.fusion_net = FusionNet(model_name, num_classes)
		self.criterion = nn.BCELoss()

	def forward(self, img, target = None):
		output_global, fm_global, pool_global = self.global_net(img)
		image_patch = self.attention_gen_patchs(img, fm_global)
		output_local, _, pool_local = self.local_net(image_patch)
		output_fusion = self.fusion_net(pool_global, pool_local)

		if target is not None:
			global_loss = self.criterion(output_global, target)
			local_loss = self.criterion(output_local, target)
			fusion_loss = self.criterion(output_fusion, target)
		else:
			global_loss = torch.tensor(0, dtype=img.dtype, device=img.device)
			local_loss = torch.tensor(0, dtype=img.dtype, device=img.device)
			fusion_loss = torch.tensor(0, dtype=img.dtype, device=img.device)

		loss = 0.8 * global_loss + 0.1 *local_loss + 0.1 * fusion_loss

		output = {
			"image_patch": image_patch,
			"global": output_global,
			"local": output_local,
			"fusion": output_fusion,
			"global_loss": global_loss,
			"local_loss": local_loss,
			"fusion_loss": fusion_loss,
			"loss": loss
		}

		return output

	def attention_gen_patchs(self, ori_image, features_global, threshold = 0.7):

		features_global = features_global.cpu()
		batch_size = features_global.shape[0]
		n, c, h, w = ori_image.shape

		cropped_image = torch.randn(batch_size, c, w, h, dtype=ori_image.dtype, device=ori_image.device)

		for b in range(batch_size):
			heatmap = torch.abs(features_global[b])
			heatmap = torch.max(heatmap, dim = 0)[0]
			max1 = torch.max(heatmap)
			min1 = torch.min(heatmap)
			heatmap = (heatmap - min1) / (max1 - min1)

			heatmap = transforms.ToPILImage()(heatmap)
			heatmap = transforms.Resize((h, w))(heatmap)
			heatmap = transforms.ToTensor()(heatmap).squeeze(0)

			heatmap[heatmap > threshold] = 1
			heatmap[heatmap != 1] = 0

			heatmap = self._select_max_connect(heatmap)

			where = torch.from_numpy(np.argwhere(heatmap == 1))
			xmin = int(torch.min(where, dim = 0)[0][0])
			xmax = int(torch.max(where, dim = 0)[0][0])
			ymin = int(torch.min(where, dim = 0)[0][1])
			ymax = int(torch.max(where, dim = 0)[0][1])

			image = transforms.ToPILImage()(ori_image[b][:, xmin:xmax, ymin:ymax].cpu())
			heatmap = transforms.Resize((h, w))(image)
			cropped_image[b] = transforms.ToTensor()(heatmap)

		return cropped_image.to(ori_image.dtype)

	def _select_max_connect(self, heatmap):
		labeled_img, num = measure.label(heatmap, connectivity = 2, background = 0, return_num = True)    
		max_label = 0
		max_num = 0
		for i in range(1, num + 1):
			if np.sum(labeled_img == i) > max_num:
				max_num = np.sum(labeled_img == i)
				max_label = i
		lcc = (labeled_img == max_label)
		if max_num == 0:
			lcc = (labeled_img == -1)
		lcc = lcc + 0
		return lcc