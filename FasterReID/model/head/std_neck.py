# -*- coding: utf-8 -*-
# @Author: solicucu

import torch 
import torch.nn as nn 

class STDNeck(nn.Module):

	def __init__(self, num_class, in_planes, dropout = 0., fc_dims = [], use_bnneck = False):

		super(STDNeck, self).__init__()

		self.fc_dims = fc_dims
		self.use_bnneck = use_bnneck
		self.final_planes = in_planes
		self.dropout = dropout
		self.fc_num = len(fc_dims)
		self.GP = nn.AdaptiveAvgPool2d(1)
		if self.fc_num > 0:
			self.fc = self._make_fc_layers(in_planes)

		if self.dropout > 0:
			self.drop = nn.Dropout(dropout)

		if self.use_bnneck:
			self.bnneck = nn.BatchNorm1d(self.final_planes)
			

		self.classifier = nn.Linear(self.final_planes, num_class)


	def _make_fc_layers(self, in_planes):

		layers = []
		in_dim = in_planes

		for i, dim in enumerate(self.fc_dims):
			layers.append(nn.Linear(in_dim, dim))
			layers.append(nn.BatchNorm1d(dim))
			layers.append(nn.ReLU(inplace = True))

			in_dim = dim 

		self.final_planes = in_dim

		return nn.Sequential(*layers)


	def forward(self, x):

		# need a featrue map before GAP
		feat = self.GP(x)
		batch = feat.size(0)
		feat = feat.view(batch, -1)

		if self.fc_num > 0:
			feat = self.fc(feat)
		# dropout
		if self.dropout > 0:
			feat = self.drop(feat)
		
		if self.use_bnneck:
			# bn_feat = torch.nn.functional.normalize(feat, dim = 1, p = 2)
			bn_feat = self.bnneck(feat)  # return for triplet loss
		else:
			bn_feat = feat 

		scores = self.classifier(feat)

		if self.training:
			# return the result in the form of list
			return [[scores, bn_feat]]
		else:
			return bn_feat 
			# return [scores] # for classification
			# for extract features 
			# return [bn_feat, feat]



