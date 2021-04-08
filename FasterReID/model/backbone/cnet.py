# -*- coding: utf-8 -*-
# @Author: solicucu

import torch 
import sys 
# sys.path.append(root) 
import json  

from factory.cnet_sample_topk_search import * 
# from config import cfg 

def make_divisible(num, divisor = 4):

	return int((num // divisor) * divisor)

class CNetwork(nn.Module):

	def __init__(self, num_class, cfg):

		super(CNetwork, self).__init__()
		self.num_class = num_class 
		self.stages = cfg.MODEL.STAGES 
		self.use_gpu = cfg.MODEL.DEVICE == "cuda"
		self.multiplier = cfg.MODEL.WIDTH_MULT 
		self.pretrained = cfg.MODEL.PRETRAIN_PATH 
		self.genotype = cfg.MODEL.GENOTYPE 
		self.adaption_fusion = cfg.MODEL.ADAPTION_FUSION 
		self.fc_dims = cfg.MODEL.FC_DIMS
		self.fc_num = len(self.fc_dims)
		self.before_gap = True
		# return each stages feat 
		self.extract_stages_feats = cfg.MODEL.NECK_TYPE == "fblneck"
		# change the channel for scaling 
		self.planes = [make_divisible(n * self.multiplier) for n in cfg.MODEL.PLANES]
		self.final_planes = 512 
		# self.final_planes = 1024

		self.stem = StdStem(3, self.planes[0], kernel_size = 7, stride = 2, padding = 3)

		# read the genotype 
		# self.genotype_file = cfg.OUTPUT.DIRS + self.genotype
		self.genotype_file = "../geno/" + self.genotype 

		with open(self.genotype_file, 'r') as f:
			self.geno = json.load(f)

		kernels = self.geno["layers"]

		self.cells = nn.ModuleList()
		 
		num = len(self.stages)
		
		for i in range(num - 1):
			kernel_list = kernels[i*2: i*2 + 2]
			self.cells += self._make_layers(self.stages[i], self.planes[i], self.planes[i+1], kernel_list)
			self.cells.append(DownSample(self.planes[i+1], self.planes[i+1]))

		self.cells += self._make_layers(self.stages[-1], self.planes[-2], self.planes[-1], kernels[-2:])
	
		# gap
		self.gap = nn.AdaptiveAvgPool2d(1)
		# fully connected layer 
		self.final_planes = self.planes[-1]
		# if is before gap no need setup fc 
		if self.fc_num > 0 and not self.before_gap:
			self.fc = self._make_fc_layers(self.planes[-1])

		if not self.before_gap:
			self.classifier = nn.Linear(self.final_planes, num_class)

	def _make_layers(self, num_cells, in_planes, out_planes, kernel_list):

		cells = []
		k1, k2 = kernel_list[0]
		cells.append(CBlock(in_planes, out_planes, k1, k2, adaptionfuse = self.adaption_fusion))
		for i in range(num_cells - 1):
			k1, k2 = kernel_list[i+1]
			cells.append(CBlock(out_planes, out_planes, k1, k2, adaptionfuse = self.adaption_fusion))

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


	def forward(self, x):
		# c1,c2,d1,c1,c2,d2,c1,c2
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
		# gap 
		x = self.gap(x)

		# reshape the shape
		batch = x.size(0)
		x = x.view(batch, -1)

		if self.fc_num > 0:
			x = self.fc(x)

		if not self.training:
			return x 

		score = self.classifier(x)

		return [[score, x]]
		





if __name__ == "__main__":

	tensor = torch.randn(2,3, 256, 128)
	# model = Cell(24, 24, [3,5], usesub = False)
	model = CNetwork(751, cfg)
	print(model)
	res = model(tensor)
	print(res.size())