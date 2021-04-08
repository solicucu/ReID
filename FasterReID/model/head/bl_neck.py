# -*- coding: utf-8 -*-
# @Author: solicucu

import torch 
import torch.nn as nn 
"""
balance neck:
use fc layer to balance triplet and fc 
"""
class BLNeck(nn.Module):

	def __init__(self, num_class, in_planes, dropout = 0., fc_dims = [512], classification = False):

		super(BLNeck, self).__init__()

		self.fc_dims = fc_dims 
		self.fc_num = len(fc_dims)
		self.final_planes = in_planes
		self.dropout = dropout 
		self.classification = classification

		self.GP = nn.AdaptiveAvgPool2d(1)
		if self.dropout > 0:
			self.drop = nn.Dropout(dropout)

		if self.fc_num > 0:
			self.fcs = self._make_fc_layers(in_planes)
			self.final_planes = self.fc_dims[-1]

		self.classifier = nn.Linear(self.final_planes, num_class)

	def _make_fc_layers(self, in_planes):

		layers = []
		in_dim = in_planes
		for i, dim in enumerate(self.fc_dims):
			layers.append(nn.Linear(in_dim, dim))
			layers.append(nn.BatchNorm1d(dim))
			layers.append(nn.ReLU(inplace = True))

			in_dim = dim 

		return nn.Sequential(*layers)


	def forward(self, x):

		# global pooling
		feat = self.GP(x)
		batch = feat.size(0)
		feat = feat.view(batch, -1)

		if self.dropout > 0:
			feat = self.drop(feat)
		
		# fc for softmax 
		if self.fc_num > 0:
			feat_fc = self.fcs(feat)
		else:
			feat_fc = feat

		scores = self.classifier(feat_fc)

		if self.training:
			return [[scores, feat]]
		else:
			if self.classification:
				return [scores]
			else:
				return feat