# -*- coding: utf-8 -*-
# @Author: solicucu

import numpy as np 
import torch
import torch.nn as nn 
from torch.optim import lr_scheduler

def make_optimizer(cfg, model):

	# for some layer was frozen 
	params = filter(lambda p: p.requires_grad, model.parameters())
	
	if cfg.SOLVER.OPTIMIZER_NAME == "SGD":

		optimizer = getattr(torch.optim, "SGD")(params, lr = cfg.SOLVER.BASE_LR, momentum = cfg.SOLVER.MOMENTUM, weight_decay = cfg.SOLVER.WEIGHT_DECAY)

	elif cfg.SOLVER.OPTIMIZER_NAME == "Adam":

		optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, lr = cfg.SOLVER.BASE_LR, weight_decay = cfg.SOLVER.WEIGHT_DECAY)

	elif cfg.SOLVER.OPTIMIZER_NAME == 'AMSGrad':

		optimizer = getattr(torch.optim, "Adam")(params, lr = cfg.SOLVER.BASE_LR, weight_decay = cfg.SOLVER.WEIGHT_DECAY, betas=(0.9, 0.99), amsgrad=True)
		
	return optimizer

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):

	def __init__(self, optimizer, milestones = [40, 70], lr_list = None, gama = 0. , warmup_factor = 0.1,
				warmup_iters = 10, warmup_method = 'linear', last_epoch = -1):

		self.milestones = milestones 
		self.lr_list = lr_list
		if gama == 0 and warmup_iters > 0:
			self.gama = 1. / warmup_iters 
		else:
			self.gama = gama  
		self.warmup_factor = warmup_factor  
		self.warmup_iters = warmup_iters 
		self.warmup_method = warmup_method
		self.stages = len(self.milestones)

		super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

	def get_lr(self):
		# self.last_epoch is current epoch
		if self.last_epoch < self.warmup_iters:
			if self.warmup_method == 'linear':
				cur_iter = self.last_epoch + 1
				# warmup_factor default is 1.
				# gama = 1/warmup_iters
				# lr = init_lr * gama * t, where init_lr = base_lr * warmup_factor
				lr = self.base_lrs[0] * self.warmup_factor * self.gama * cur_iter
				return [lr]
			else:
				raise NotImplementedError("not know such warmup method {}".format(self.warmup_method) )

		else:
			# default is the first lr
			lr = self.lr_list[0]
			for i in range(self.stages):
				# select the proper lr
				if self.last_epoch >= self.milestones[i]:
					lr = self.lr_list[i+1]

			return [lr]


# torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)
class WarmupCosAnnLR(torch.optim.lr_scheduler._LRScheduler):
	# T_max is END_EPOCH
	def __init__(self, optimizer, T_max, eta_min, last_epoch = -1, gama = 0., warmup_factor = 0.1,
				warmup_iters = 10, warmup_method = 'linear'):
		
		self.eta_min = eta_min
		self.T_max = T_max - warmup_iters
		if gama == 0:
			self.gama = 1. / warmup_iters 
		else:
			self.gama = gama  
		self.warmup_factor = warmup_factor  
		self.warmup_iters = warmup_iters 
		self.warmup_method = warmup_method
		# for constant warmup 
		# warmup iter include two part, constant and linear 
		self.first_warmup = warmup_iters - 10
		
		super(WarmupCosAnnLR, self).__init__(optimizer, last_epoch)
		

	def get_lr(self):

		eta_max_sub_min = self.base_lrs[0] - self.eta_min

		if self.last_epoch < self.warmup_iters:
			if self.warmup_method == 'linear':

				cur_iter = self.last_epoch + 1
				# lr = init_lr * gama * t, where init_lr = base_lr * warmup_factor
				lr = self.base_lrs[0] * self.warmup_factor * self.gama * cur_iter
				return [lr] 
			elif self.warmup_method == 'constant':
				if self.last_epoch < self.first_warmup:
				# first parts
				# constant will return a constant lr as follow
					lr = self.base_lrs[0] * self.warmup_factor
					return [lr]
				# second part use linear method to smooth 
				else:
					cur_iter = self.last_epoch + 1
					lr = self.base_lrs[0] * self.warmup_factor * self.gama * (cur_iter - self.first_warmup)
					return [lr]
			else:
				raise NotImplementedError("not know warmup method {}".format(self.warmup_method))	            
		else:
			cur_iter = self.last_epoch - self.warmup_iters
			lr = self.eta_min + 0.5 * eta_max_sub_min * (1 + np.cos((cur_iter / self.T_max) * np.pi))
			return [lr]
		


def make_lr_scheduler(cfg, optimizer):

	name = cfg.SOLVER.LR_SCHEDULER_NAME
	if  name == "StepLR":

		scheduler = lr_scheduler.StepLR(optimizer, step_size = cfg.SOLVER.LR_DECAY_PERIOD, gamma = cfg.SOLVER.LR_DECAY_FACTOR)
	
	elif name == "CosineAnnealingLR":

		scheduler = lr_scheduler.CosineAnnealingLR(optimizer, float(cfg.SOLVER.MAX_EPOCHS), eta_min = cfg.SOLVER.LR_MIN)

	elif name == "WarmupMultiStepLR":

		scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.MILESTONES, cfg.SOLVER.LR_LIST, cfg.SOLVER.GAMA, cfg.SOLVER.WARMUP_FACTOR, cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

	elif name == "WarmupCosAnnLR":

		scheduler = WarmupCosAnnLR(optimizer, cfg.SOLVER.MAX_EPOCHS, cfg.SOLVER.LR_MIN, gama = cfg.SOLVER.GAMA, warmup_factor = cfg.SOLVER.WARMUP_FACTOR,
					warmup_iters = cfg.SOLVER.WARMUP_ITERS, warmup_method = cfg.SOLVER.WARMUP_METHOD)
		
	else:

		raise RuntimeError(" name {} is not know".format(name))

	return scheduler 	
