# -*- coding: utf-8 -*-
# @Author: solicucu

import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import json 
import copy 
import glob 
import logging 
import os 
import torch.nn.init as init 

from operations import * 
from genotypes import * 
from torch.autograd import Variable


logger = logging.getLogger("CNet_Search.cnet")
class MBlock(nn.Module):

	def __init__(self, in_planes, out_planes,  adaptionfuse = False, usek9 = True):
		super(MBlock, self).__init__()

		if not usek9:
			kernel = kernels[:-3]
		else:
			kernel = kernels

		self.ops = nn.ModuleList()
		
		for k1,k2 in kernel:
			cblock = CBlock(in_planes, out_planes, k1, k2, adaptionfuse = adaptionfuse)
			self.ops.append(cblock)

	def forward(self, x, weights):

		return sum( w * op(x) for w, op in zip(weights, self.ops))
		

def make_divisible(num, divisor = 4):

	return int((num // divisor) * divisor)

class CNetwork(nn.Module):
	"""
	Params:
		num_class: the number of class to be classified
		planes: the base channel for each stage
		layers: the number of repeated block for each stages
		multiplier: a float number used to scale the width of the network range(1.,3.)
		use_gpu: whether use gpu to train the network
		pretrained: path to the pretained checkpoint
	Returns:
		score: clasification score
		feat: feature for computing triplet loss
	"""
	def __init__(self, num_class, cfg):

		super(CNetwork, self).__init__()
		self.num_class = num_class
		self.stages = cfg.MODEL.STAGES
		self.use_gpu = cfg.MODEL.DEVICE == "cuda"
		self.multiplier = cfg.MODEL.MULTIPLIER 
		self.cfg = cfg
		self.pretrained = cfg.MODEL.PRETRAINED 
		self.adaption_fusion = cfg.MODEL.ADAPTION_FUSION
		self.fc_dims = cfg.MODEL.FC_DIMS
		self.fc_num = len(self.fc_dims)
		# change the channel for scaling
		self.planes = [make_divisible(n * self.multiplier) for n in cfg.MODEL.PLANES]
		self.final_planes = 512
		self.before_gap = False
		self.dropout = 0.2 


		self.stem = StdStem(3, self.planes[0], kernel_size = 7, stride = 2, padding = 3)
		
		self.cells = nn.ModuleList()

		num = len(self.stages)
		for i in range(num - 1):
			self.cells += self._make_layers(self.stages[i],self.planes[i], self.planes[i+1], usek9 = True)
			self.cells.append(DownSample(self.planes[i+1], self.planes[i+1]))

		self.cells += self._make_layers(self.stages[-1],self.planes[-2], self.planes[-1], usek9 = True)

		# expand or squeeze the channel
		self.conv1x1 = Conv1x1BNReLU(self.planes[-1], self.planes[-1])
		# change the feature map into vector
		
		# gap
		self.gap = nn.AdaptiveAvgPool2d(1)
		# fully connected layer 
		self.final_planes = self.planes[-1]

		# balance neck
		if self.fc_num > 0:
			self.fc = self._make_fc_layers(self.planes[-1])
			self.final_planes = self.fc_dims[-1]

		if self.dropout > 0:
			self.drop = nn.Dropout(self.dropout)

		# classifier
		if not self.before_gap:
			self.classifier = nn.Linear(self.final_planes, num_class)

		self._init_alphas()


	def _make_layers(self, num_cells, in_planes, out_planes, usek9 = True):

		cells = []
		cells.append(MBlock(in_planes, out_planes, self.adaption_fusion, usek9))
		for i in range(num_cells - 1):
			cells.append(MBlock(out_planes, out_planes, self.adaption_fusion, usek9))

		return cells

	def _make_fc_layers(self, in_planes):

		layers = []
		in_dim = in_planes

		for dim in self.fc_dims:

			layers.append(nn.Linear(in_dim, dim))
			layers.append(nn.BatchNorm1d(dim))
			layers.append(nn.ReLU(inplace = True))

			in_dim = dim 

		return nn.Sequential(*layers)

	def _init_alphas(self):
		# number of layers in the frist three stage
		k1 = sum(self.stages)

		num_ops = len(op_names)
		scale = 1e-3
		if self.use_gpu:
			self.alphas1 = Variable(scale * torch.ones(k1, num_ops).cuda(), requires_grad = True)

		else:
			self.alphas1 = Variable(scale * torch.ones(k1, num_ops), requires_grad = True)

		self.arch_parameters = [self.alphas1]

	
	def forward(self, x):
		
		x = self.stem(x)
		
		pos = -1

		weights1 = F.softmax(self.alphas1, dim = -1)
		new_stage = self.stages[:-1]

		#1~2 stages
		w = -1 
		for i, num in enumerate(new_stage):
			# each stage
			for j in range(num):
				pos += 1
				w += 1
				x = self.cells[pos](x, weights1[w])
			# downsample
			pos += 1
			
			x = self.cells[pos](x)

		# last stage, without downsample
		for i in range(self.stages[-1]):
			pos += 1
			w += 1
			x = self.cells[pos](x, weights1[w])
		
		# conv1x1 before gap
		x = self.conv1x1(x)

		if self.before_gap:
			return x 
		# gap 
		x = self.gap(x)

		# reshape the shape
		batch = x.size(0)
		feat = x.view(batch, -1)

		# dropout 
		if self.dropout > 0:
			feat = self.drop(feat)

		if self.fc_num > 0:
			feat_fc = self.fc(feat)
		else:
			feat_fc = feat 

		if not self.training:
			return feat

		score = self.classifier(feat_fc)

		return [[score, feat]]

	def _arch_parameters(self):

		return self.arch_parameters 

	def new(self):

		model_new = CNetwork(self.num_class, self.cfg)

		if self.use_gpu:
			model_new = model_new.cuda()

		for x, y in zip(model_new._arch_parameters(), self._arch_parameters()):
			x.data.copy_(y.data)

		model_new._set_loss(self.loss_fn, self.loss_acc_fn)

		return model_new 

	def _set_loss(self, loss_fn, loss_acc_fn):

		self.loss_fn = loss_fn 
		self.loss_acc_fn = loss_acc_fn 

	def _loss(self, imgs, labels):

		res = self(imgs)

		loss, _ = self.loss_acc_fn(res, labels, self.loss_fn)

		return loss  

	def _parse_genotype(self, file = "./genotype.json"):

		geno = {}

		w1 = self.alphas1
		# find the maxvlaue indices
		_, indices1 = w1.max(dim = -1)

		layers = []

		if self.use_gpu:
			indices1 = indices1.cpu().numpy()
		else:
			indices1 = indices1.numpy()

		for ind in indices1:
			layers.append(kernels[ind])

		geno["layers"] = layers

		# alphas
		alphas1 = copy.deepcopy(self.alphas1)

		if self.use_gpu:
			alphas1 = alphas1.cpu()

		alphas1 = alphas1.detach().numpy().tolist()

		geno["alphas1"] = alphas1

		json_data = json.dumps(geno, indent = 4)
		with open(file, 'w') as f:
			f.write(json_data)

		# return geno  

	def kaiming_init_(self):

		# print("use kaiming init")
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				init.kaiming_normal_(m.weight)
				if m.bias is not None:
					init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				if m.weight is not None:
					init.constant_(m.weight, 1)
				if m.bias is not None:
					init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				init.normal_(m.weight, std = 1e-3)
				if m.bias is not None:
					init.constant_(m.bias, 0)

	def load_pretrained_model(self, path):
		
		# checkpoint
		ckpt_list = glob.glob(path + "checkpoint_*")
		# 
		ckpt_list = sorted(ckpt_list)

		ckpt_name = ckpt_list[-1]
		"""
		file_path = "D:/test/test.py"
		(filepath, tempfilename) = os.path.split(file_path)
		(filename, extension) = os.path.splitext(tempfilename)
		"""
		num = int(os.path.split(ckpt_name)[1].split("_")[1].split(".")[0])
		self.start_epoch = num
		logger.info("load checkpoint from {}".format(ckpt_name))

		self.load_state_dict(torch.load(ckpt_name))

		# genotype
		geno_name = path + "genotype_{}.json".format(num)

		with open(geno_name, 'r') as f:
			geno = json.load(f)

		alphas1 = torch.tensor(geno["alphas1"])

		self.alphas1.data.copy_(alphas1)

		logger.info("end of load the checkpoint and alpha")


if __name__ == "__main__":
	
	tensor = torch.randn(2, 3, 256, 128)
	weights = [1.,1.,1.,1.,1.,1.]
	# print(np.prod(torch.randn(1,1,192, 512).size())/ 1e6)
	# exit(1)
	model = CNetwork(1000)
	# print(model)
