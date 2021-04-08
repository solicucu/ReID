# -*- coding: utf-8 -*-
# @Author: solicucu

import torch
import torch.nn as nn 
import torch.nn.functional as F 

# no relu or batch norm follow it
def conv1x1(in_planes, out_planes):
	return nn.Conv2d(in_planes, out_planes, kernel_size = 1, padding = 0, bias = False)

# conv1x1 followed by bn and relu
class Conv1x1BNReLU(nn.Module):

	def __init__(self, in_planes, out_planes):

		super(Conv1x1BNReLU, self).__init__()

		self.op = nn.Sequential(
			nn.Conv2d(in_planes, out_planes, kernel_size = 1, padding = 0, bias = False),
			nn.BatchNorm2d(out_planes),
			nn.ReLU(inplace = True)
		)

	def forward(self, x):

		return self.op(x)

# conv1x1 with bn no relu for linear transformation

class Conv1x1BN(nn.Module):

	def __init__(self, in_planes, out_planes):

		super(Conv1x1BN, self).__init__()

		self.op = nn.Sequential(
			nn.Conv2d(in_planes, out_planes, kernel_size = 1, padding = 0, bias = False),
			nn.BatchNorm2d(out_planes)
		)

	def forward(self, x):

		return self.op(x)

"""
standard convolution 3x3, 5x5
"""
class ConvBNReLU(nn.Module):

	def __init__(self, in_planes, out_planes, kernel_size = 3, stride = 1, padding = 1, keepsame = True):

		super(ConvBNReLU, self).__init__()
		# resize the padding to keep the same shape for special kernel_size
		if keepsame:
			padding = kernel_size // 2
		self.op = nn.Sequential(
			nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = stride, padding = padding, bias = False),
			nn.BatchNorm2d(out_planes),
			nn.ReLU(inplace = True)
		)

	def forward(self, x):

		return self.op(x)

class AdaptiveFuse(nn.Module):

	def __init__(self, in_planes, reduction = 4, layer_norm = False):

		super(AdaptiveFuse, self).__init__()
		self.layer_norm = layer_norm 
		mid_planes = in_planes // reduction
		self.gap = nn.AdaptiveAvgPool2d(1)
		self.fc1 = nn.Conv2d(in_planes, mid_planes, kernel_size = 1, padding = 0, bias = True)
		if self.layer_norm:
			self.norm = nn.LayerNorm(mid_planes)
		self.relu = nn.ReLU(inplace = True)
		self.fc2 = nn.Conv2d(mid_planes, in_planes, kernel_size = 1, padding = 0, bias = True)
		self.activation = nn.Sigmoid()

	def forward(self, x):

		res = self.gap(x)
		res = self.fc1(res)
		if self.layer_norm:
			res = self.norm(res)
		res = self.relu(res)
		res = self.fc2(res)

		w = self.activation(res)

		return x * w 


class StdStem(nn.Module):

	def __init__(self, in_planes, out_planes, kernel_size, stride, padding, usepool = True):

		super(StdStem, self).__init__()
		self.usepool = usepool
		self.conv1 = ConvBNReLU(in_planes, out_planes, kernel_size, stride, padding, keepsame = False)
		if self.usepool:
			self.pool = nn.MaxPool2d(3, stride=2, padding=1)
# 
	def forward(self, x):

		x = self.conv1(x)
		if self.usepool:
			return self.pool(x)
		return x 


class DWBlock(nn.Module):
	# notice that in dwblock, in_planes always equal to out_planes
	# we reverse the order of conv1x1 and dw_conv as osnet
	def __init__(self, in_planes, out_planes, kernel_size = 3, stride = 1, padding = 1, keepsame = True):

		super(DWBlock, self).__init__()
		assert in_planes == out_planes
		if keepsame:
			padding = kernel_size // 2
		self.op = nn.Sequential(
			conv1x1(in_planes, out_planes),
			nn.Conv2d(out_planes, out_planes, kernel_size = kernel_size, stride = stride, padding = padding, groups = out_planes, bias = False),
			nn.BatchNorm2d(out_planes),
			nn.ReLU(inplace = True)
		)

	def forward(self, x):

		return self.op(x)
"""
Combined Block: combine two type kernel together
we will keep the same shape and the stride always is 1 
param: 
	k1,k2 is two kernel_size respectively
return:
	res = res1 + res2 
	
"""
class CBlock(nn.Module):

	def __init__(self, in_planes, out_planes, k1, k2, reduction = 4, adaptionfuse = False):

		super(CBlock, self).__init__()
		
		self.adaptionfuse = adaptionfuse
		mid_planes = out_planes // reduction 

		self.squeeze = Conv1x1BNReLU(in_planes, mid_planes)
		self.conv1 = self.make_block_layer(mid_planes, k1)
		self.conv2 = self.make_block_layer(mid_planes, k2)
		if self.adaptionfuse:
			self.adaption = AdaptiveFuse(mid_planes)


		# note the conv1x1 is linear
		self.restore = Conv1x1BN(mid_planes, out_planes)

		# use for identity
		self.expand = None
		# note the conv1x1 is linear
		if in_planes != out_planes:
			self.expand = Conv1x1BN(in_planes, out_planes)

		self.relu = nn.ReLU(inplace = True)

	def forward(self, x):

		identity = x 

		# reduction for bottleneck
		x = self.squeeze(x)
		res1 = self.conv1(x)
		res2 = self.conv2(x)

		if self.adaptionfuse:
			add = self.adaption(res1) + self.adaption(res2)
		else:
			add = res1 + res2


		res = self.restore(add)

		if self.expand is not None:

			identity = self.expand(identity)

		out = res + identity 

		return self.relu(out)
	

	def make_block_layer(self, in_planes, k):

		blocks = []
		# compute how many dw_conv3x3 to be construct
		num = k // 2
		for i in range(num):
			blocks.append(DWBlock(in_planes, in_planes))

		return nn.Sequential(*blocks)

"""
Param:
	kernel_list: a 3-elem list [k1, k2, r], r denote the number of cblock to be constructed
	in_planes will be same with out_planes, channel only change in downsample 
"""
class CDBlock(nn.Module):

	def __init__(self, in_planes, out_planes, kernel_list, adaptionfuse = False):

		super(CDBlock, self).__init__()
		blocks = []
		k1, k2, r = kernel_list
		# first block may occur expand channel
		blocks.append(CBlock(in_planes, out_planes, k1, k2,  adaptionfuse = adaptionfuse))
		for i in range(r-1):
			blocks.append(CBlock(out_planes, out_planes, k1, k2, adaptionfuse = adaptionfuse))	

		self.ops = nn.Sequential(*blocks)

	def forward(self, x):

		return self.ops(x)
		

class DownSample(nn.Module):

	def __init__(self, in_planes, out_planes, stride = 2):

		super(DownSample, self).__init__()
		self.avg_pool = nn.AvgPool2d(2, stride = stride, padding = 0)
		self.conv1x1 = Conv1x1BNReLU(in_planes, out_planes)
	# here we do conv1x1 first before avg_pool as osnet
	def forward(self, x):
		x = self.conv1x1(x)
		return  self.avg_pool(x)


if __name__ == "__main__":

	# tensor = torch.randn(1,3, 256, 128)
	tensor = torch.randn(1, 3, 8, 4)
	# model = ConvBNReLU(3,12, kernel_size = 9, stride = 2, padding = 4)
	# model = StdMixedBlock(3, 12)
	# model = DWBlock(3,12)
	# model = CBlock(3,3,5,9)
	# model = DownSample(3, 6)
	# print(model)
	x = model(tensor, [1.,1.,1.])
	# x = model(tensor)
	print(x.size())
