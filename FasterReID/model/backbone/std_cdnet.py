# -*- coding: utf-8 -*-
# @Author: solicucu

import torch 
import torch.nn as nn 
import json  
import sys 
from factory.cdnet_sample_topk_search import *

class STDBlock(nn.Module):

	def __init__(self, in_planes, out_planes,k1, k2, reduction = 4):

		super(STDBlock, self).__init__()
		mid_planes = out_planes // reduction 

		self.squeeze = Conv1x1BNReLU(in_planes, mid_planes)
		self.conv1 = self.make_layers(mid_planes, k1)
		self.conv2 = self.make_layers(mid_planes, k2)
		self.restore = Conv1x1BN(mid_planes, out_planes)

		self.expand = None 

		if in_planes != out_planes:
			self.expand = Conv1x1BN(in_planes, out_planes)
		
		self.relu = nn.ReLU(inplace = True)

	def forward(self, x):
		
		identity = x 
		x = self.squeeze(x)
		x  = self.conv1(x)
		x = self.conv2(x)
		x = self.restore(x)

		if self.expand is not None:

			identity = self.expand(identity)

		out = x + identity 

		return self.relu(out)

	def make_layers(self, in_planes, k):

		layers = [] 
		num = k // 2
		for i in range(num):
			layers.append(DWBlock(in_planes, in_planes))
		return nn.Sequential(*layers)

class STDDBlock(nn.Module):

	def __init__(self, in_planes, out_planes, kernel_list):

		super(STDDBlock, self).__init__()
		blocks = []
		k1, k2, r = kernel_list
		# first block may occur expand channel
		blocks.append(STDBlock(in_planes, out_planes, k1, k2))
		for i in range(r-1):
			blocks.append(STDBlock(out_planes, out_planes, k1, k2))	

		self.ops = nn.Sequential(*blocks)

	def forward(self, x):

		return self.ops(x)

def make_divisible(num, divisor = 4):

	return int((num // divisor) * divisor)


class STDNetwork(nn.Module):

	def __init__(self, num_class, cfg):

		super(STDNetwork, self).__init__()
		self.num_class = num_class 
		self.stages = cfg.MODEL.STAGES 
		self.use_gpu = cfg.MODEL.DEVICE == "cuda"
		self.multiplier = cfg.MODEL.WIDTH_MULT 
		self.pretrained = cfg.MODEL.PRETRAIN_PATH 
		self.genotype = cfg.MODEL.GENOTYPE 
		self.before_gap = True

		# change the channel for scaling 
		self.planes = [make_divisible(n * self.multiplier) for n in cfg.MODEL.PLANES]
		self.final_planes = 512 
		self.extract_stages_feats = cfg.MODEL.NECK_TYPE == 'fblneck'

		self.stem = StdStem(3, self.planes[0], kernel_size = 7, stride = 2, padding = 3)

		# read the genotype 
		# self.genotype_file = cfg.OUTPUT.DIRS + self.genotype
		self.genotype_file = "../geno/" + self.genotype 

		with open(self.genotype_file, 'r') as f:
			self.geno = json.load(f)

		kernels = self.geno["layers"]

		self.cells = nn.ModuleList()
		# before last stage 
		num = len(self.stages)

		for i in range(num - 1):
			kernel_list = kernels[i*2: i*2 + 2]
			self.cells += self._make_layers(self.stages[i], self.planes[i], self.planes[i+1], kernel_list)
			self.cells.append(DownSample(self.planes[i+1], self.planes[i+1]))

		self.cells += self._make_layers(self.stages[-1], self.planes[-2], self.planes[-1], kernels[-2:])

	def _make_layers(self, num_cells, in_planes, out_planes, kernel_list):

		cells = []
	
		cells.append(STDDBlock(in_planes, out_planes, kernel_list[0]))
		for i in range(num_cells - 1):
			cells.append(STDDBlock(out_planes, out_planes, kernel_list[i+1]))

		return cells 

	def forward(self, x):

		x = self.stem(x)
		
		stages = [3, 6, 8 ]
		i = 0 
		feature_maps = []
		for cell in self.cells:

			x =  cell(x)
			i += 1
			if self.extract_stages_feats and i in stages:
				feature_maps.append(x)

		# return the feature and no need to return below x 
		if self.extract_stages_feats:
			return feature_maps
			
		if self.before_gap:
			return x 