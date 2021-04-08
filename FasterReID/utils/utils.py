# -*- coding: utf-8 -*-
# @Author: solicucu

import numpy as np
import json
import torch 
import copy 
import re 

"""
BatchNorm2d: has weight and bias -> default the affine = True
"""
def count_parameters(model):
	# print(model)
	# for name, param in model.named_parameters():
	# 	print(name, param.size())
		
	return np.sum(np.prod(param.size()) for param in model.parameters()) / 1e6

# count the parameter rm the fc of blneck and classifier
def infer_count_parameters(model):
	# only in fblneck fc named fcs 
	# removes = ['fcs', 'classifier']
	params = []
	for name, param in model.named_parameters():
		if 'fcs' in name or 'classifier' in name:
			continue 
		params.append(np.prod(param.size()))
		
	params = np.array(params)
	return np.sum(params) / 1e6

# only param in keys set requires_grad = True ,other is False
def frozen_some_layers(keys, model):
	# 
	for name, param in model.named_parameters():
		param.grad = False
		for key in keys:
			if key in name:
				param.grad = True  
				break 
	return model 