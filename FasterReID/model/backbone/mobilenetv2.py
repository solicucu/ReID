# -*- coding: utf-8 -*-
# @Author: solicucu

import torch 
from torch import nn 
from torch.utils.model_zoo import load_url 

model_url = "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth"

# input_channel * width_mult, round_nearest
# note that operator // is compute first than *

def _make_divisible(v, divisor, min_value = None):

	if min_value is None:
		min_value = divisor
	# v + divisor/2 is for upper round
	new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
	# Make sure that round down does not go down by more than 10%.
	if new_v < 0.9 * v:
		new_v += divisor
	return new_v 

class ConvBNReLU(nn.Sequential):

	def __init__(self, in_planes, out_planes, kernel_size = 3, stride = 1, group = 1):

		padding = (kernel_size - 1) // 2
		super(ConvBNReLU, self).__init__(
			nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups = group, bias = False),
			nn.BatchNorm2d(out_planes),
			nn.ReLU6(inplace = True)
		)

class InvertedResidual(nn.Module):

	def __init__(self,in_planes, out_planes, stride, exapned_ratio):

		super(InvertedResidual, self).__init__()
		self.stride = stride
		assert stride in [1, 2]

		hidden_dim = int(round(in_planes * exapned_ratio))
		self.use_res_connect = self.stride == 1 and in_planes == out_planes

		layers = []
		if exapned_ratio != 1:
			# pw
			layers.append(ConvBNReLU(in_planes, hidden_dim, kernel_size = 1))
		layers.extend([
			# dw
			ConvBNReLU(hidden_dim, hidden_dim, stride = stride, group = hidden_dim),
			# pw - linear
			nn.Conv2d(hidden_dim, out_planes, 1, 1, 0, bias = False),
			nn.BatchNorm2d(out_planes)
		])

		self.conv = nn.Sequential(*layers)

	def forward(self, x):

		if self.use_res_connect:
			return x + self.conv(x)
		else:
			return self.conv(x)
"""
MobileNet V2 main class

Args:
    width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
    inverted_residual_setting: Network structure
    round_nearest (int): Round the number of channels in each layer to be a multiple of this number
    Set to 1 to turn off rounding
    block: Module specifying inverted residual building block for mobilenet

"""
class MobileNetV2(nn.Module):

	def __init__(self,
				 width_mult = 1.0,
				 inverted_residual_setting = None,
				 round_nearest = 8,
				 block = None,
				 pretrained = False,
				 before_gap = False):
		super(MobileNetV2, self).__init__()

		self.pretrained = pretrained
		self.is_before_gap = before_gap

		if block is None:
			block = InvertedResidual 
		input_channel = 32
		last_channel = 1024
		self.planes = last_channel 

		if inverted_residual_setting is None:
			inverted_residual_setting = [
				# t, c, n, s
                # expansion out_channel, repeated number, stride 
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
			]

		# check the inverted residual setting
		if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
			raise ValueError("inverted_residual_setting can not be None and each item must has four elem, got {}".format(inverted_residual_setting))

		# build first layer
		# default is 32, can be ajust by width_multi
		input_channel = _make_divisible(input_channel * width_mult, round_nearest)

		self.last_channel = _make_divisible(last_channel, round_nearest)
		# self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
		self.final_planes = self.last_channel 
		
		features = [ConvBNReLU(3, input_channel, stride = 2)]

		# build inverted redidual
		# t: expansion_ratio, c: out_planes, n: repeated number, s: stride
		for t, c, n, s in inverted_residual_setting:
			output_channel = _make_divisible(c * width_mult, round_nearest)
			for i in range(n):
				stride = s if i == 0 else 1
				features.append(block(input_channel, output_channel, stride, exapned_ratio = t))
				input_channel = output_channel

		# building last layer
		features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size = 1))

		self.features = nn.Sequential(*features)
		# self.GAP = nn.AvgPool2d((8,4))
		self.GAP = nn.AdaptiveAvgPool2d(1)

		if self.pretrained:
			print("use pretrained model from imagenet")
			self.load_state_dicts()

	def _forward_impl(self, x):

		x = self.features(x)
		if self.is_before_gap:
			return x
		else:
			return self.GAP(x)

	def forward(self, x):

		return self._forward_impl(x)

	def load_state_dicts(self):

		state_dicts = load_url(model_url, progress = True)
		self_state_dicts = self.state_dict()

		for key in self_state_dicts:
			self_state_dicts[key] = state_dicts[key]

		

if __name__ == "__main__":

	# model = InvertedResidual(1,2,1,6)
	imgs = torch.randn(1,3,256,128)
	model = MobileNetV2(pretrained = True, before_gap = True)
	# print(model)
	res = model(imgs)
	print(res.size()) # torch.Size([1, 1280, 8, 4])
