# -*- coding: utf-8 -*-
# @Author: solicucu

"""
fbl_neck: Fined-grain partition and BaLance neck
	we will utilize last two stage feature maps, one is for fine-grain partition,
	another is for global features
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
		
		###############################################
		# for local feat use feature map stage2 
		self.part_num = 2

		self.local_plane = planes[-2]
		# squeeze the channe into 128 
		self.mid_plane = 128 

		self.squeeze = nn.Conv2d(self.local_plane, self.mid_plane, kernel_size = 1, padding = 0, bias = False)


		###############################################
		# for global feat  #
		self.GAP = nn.AdaptiveAvgPool2d(1)

		if self.dropout > 0:
			self.drop = nn.Dropout(dropout)

		if self.fc_num > 0:
			in_planes = self.mid_plane * self.part_num + self.final_planes 
			
			# self.fc = self._make_fc_layers(in_planes)
			self.local_fcs = self._make_fc_layers(self.mid_plane * self.part_num)
			self.global_fcs = self._make_fc_layers(self.final_planes)

			self.final_planes = in_planes

		# self.classifier = nn.Linear(fc_dims[-1], num_class)
		# 
		self.local_classifier = nn.Linear(fc_dims[-1], num_class)
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

		fmap2 = x[-2]
		fmap3 = x[-1]

		#################### fine-grained partition
		fmap2 = self.squeeze(fmap2)
		parts = fmap2.chunk(self.part_num, dim = 2)

		batch = fmap2.size(0)

		# gap 
		parts = [self.GAP(p) for p in parts]
		# reshape
		local_feats = [p.view(batch, -1) for p in parts]

		##################################### local features
		# cat the feats
		local_feats = torch.cat(local_feats, dim = -1)

		################### global feature

		global_feat = self.GAP(fmap3)
		global_feat = global_feat.view(batch, -1)

		if self.dropout > 0:
			local_feats = self.drop(local_feats)
			global_feat = self.drop(global_feat)

		# feat_fc for fc
		# no need for inference 
		if self.training or self.classification:
			if self.fc_num > 0:
				global_feat_fc = self.global_fcs(global_feat)
				local_feats_fc = self.local_fcs(local_feats)
			else:
				global_feat_fc = global_feat
				local_feats_fc = local_feats

			local_scores = self.local_classifier(local_feats_fc)
			global_scores = self.global_classifier(global_feat_fc)

		if self.training:
			return [[global_scores, global_feat],[local_scores, local_feats]]
		else:
			if self.classification:
				# for imagenet, cifar 
				# return [local_scores]
				# return [global_scores]
				return [local_scores + global_scores]
			else:
				return torch.cat([global_feat, local_feats], dim = -1)
			# for extract features 
			# return [global_feat, global_feat_fc]

