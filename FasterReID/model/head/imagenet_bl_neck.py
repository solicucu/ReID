# -*- coding: utf-8 -*-
# @Author: solicucu

"""
fbl_neck: 
	here we only use global features for classification
	feature_maps:  a list of feature maps of each stages
	planes: a list of channel number 
"""
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class FBLNeck(nn.Module):

	def __init__(self, num_class,  planes, dropout = 0., fc_dims = [512], classification = False):

		super(FBLNeck, self).__init__()

		self.fc_dims = fc_dims 
		self.fc_num = len(fc_dims)
		self.final_planes = planes[-1]
		self.dropout = dropout 
		self.classification = classification
		
		# for global feat  #
		self.GAP = nn.AdaptiveAvgPool2d(1)

		if self.dropout > 0:
			self.drop = nn.Dropout(dropout)

		if self.fc_num > 0:
			self.global_fcs = self._make_fc_layers(self.final_planes)

			self.final_planes = fc_dims[-1]

		self.global_classifier = nn.Linear(fc_dims[-1], num_class)

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

		####### extract the fmap 
		# return x 

		fmap3 = x[-1]
		################### global feature
		batch = fmap3.size(0)
		global_feat = self.GAP(fmap3)
		global_feat = global_feat.view(batch, -1)

		if self.dropout > 0:
			global_feat = self.drop(global_feat)

		# feat_fc for fc
		if self.fc_num > 0:
			global_feat_fc = self.global_fcs(global_feat)
		else:
			global_feat_fc = global_feat


		global_scores = self.global_classifier(global_feat_fc)

		if self.training:
			return [[global_scores, global_feat]]
		else:
			if self.classification:
				# for imagenet, cifar 
				return [global_scores]	
			else:
				return global_feat
			

