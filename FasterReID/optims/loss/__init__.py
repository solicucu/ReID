# -*- coding: utf-8 -*-
# @Author: solicucu

import torch.nn.functional as F

from .triplet_loss import TripletLoss 
from .triplet_loss import CrossEntropyLabelSmooth

def make_loss(cfg, num_class):

	triplet = TripletLoss(cfg.SOLVER.TRI_MARGIN)
	use_gpu = cfg.MODEL.DEVICE == 'cuda'
	smooth_on = cfg.TRICKS.LABEL_SMOOTH
	# weight = cfg.SOLVER.TRIPLET_LOSS_WEIGHT 

	if smooth_on:
		smooth = CrossEntropyLabelSmooth(num_class, use_gpu = use_gpu)
		print("use CrossEntropyLabelSmooth replace with CrossEntropy")

	loss_name = cfg.SOLVER.LOSS_NAME

	if loss_name == "softmax":

		def loss_func(score, feat, target):
			#just compute the cross_entropy
			return F.cross_entropy(score, target)
	elif loss_name == "triplet":

		def loss_func(score, feat, target):

			return triplet(feat, target)[0]

	elif loss_name == "softmax_triplet":

		def loss_func(score, feat, target):

			if smooth_on:
				return  smooth(score, target) + triplet(feat, target)[0]
				
			else:
				return F.cross_entropy(score, target) +  triplet(feat, target)[0]
	else:

		raise RuntimeError("loss name:{} is not know".format(loss_name)) 

	return loss_func

def darts_make_loss(cfg):

	triplet = TripletLoss(cfg.SOLVER.TRI_MARGIN)
	# weight = 0.5 
	loss_name = cfg.SOLVER.LOSS_NAME

	if loss_name == "softmax":

		def loss_func(score, feat, target):
			#just compute the cross_entropy
			return F.cross_entropy(score, target)
	elif loss_name == "triplet":

		def loss_func(score, feat, target):

			return triplet(feat, target)[0]

	elif loss_name == "softmax_triplet":

		def loss_func(score, feat, target):

			return F.cross_entropy(score, target) +  triplet(feat, target)[0]
	else:

		raise RuntimeError("loss name:{} is not know".format(loss_name)) 

	return loss_func
	