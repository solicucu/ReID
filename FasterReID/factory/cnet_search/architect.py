# -*- coding: utf-8 -*-
# @Author: solicucu

import torch
import numpy as np 
import torch.nn as nn 
import logging 
from torch.autograd import Variable
from torch.optim import lr_scheduler

logger = logging.getLogger("CNet_Search.architect")
#combine all tensor with shape [1xn]
def concat(tensors):
	return torch.cat([x.view(-1) for x in tensors])

class Architect(object):

	def __init__(self, model, args):
		self.network_momentum = args.SOLVER.MOMENTUM 
		self.network_weight_decay = args.SOLVER.WEIGHT_DECAY 
		self.model = model
		self.gpu = args.MODEL.DEVICE == 'cuda'
		self.optimizer = torch.optim.Adam(self.model._arch_parameters(),
			lr = args.SOLVER.ARCH_LR, betas = (0.5, 0.999), weight_decay = args.SOLVER.ARCH_WEIGHT_DECAY)
		self.lr_scheduler = lr_scheduler.MultiStepLR(self.optimizer, [40, 80], 0.1)
	"""
	process of update gradient
	param: w 
	(1)no momentum and weight_decay
	   dw = dw + w * weight_decay # w * weight_decay is a regular item in order to advoid overfitting 
	   new_w = w - lr * dw
	(2)with momentum and weight_decay
	   dw = dw + dw * weight_decay
	   new_v = (dw + v * momentum), where v is previous momentum value
	   new_w = w - lr * new_v 
	"""
	def step(self, train_input, train_label, val_input, val_label, eta, network_optimizer, unrolled):
		self.optimizer.zero_grad()

		if unrolled:
			self._backward_step_unrolled(train_input, train_label, val_input, val_label, eta, network_optimizer) 
		else:
			# no unrolled, compute dalpha directly
			self._backward_step(val_input, val_label)

		self.optimizer.step()

	def _backward_step_unrolled(self,train_input, train_label, val_input, val_label, eta, network_optimizer):
		# formula 6: dαLval(w',α) ，where w' = w − ξ*dwLtrain(w, α)
		# in other words, compute dα after the model is update by (train_input, train_label)

		# create a new model to update w, the old model will be used later
		unrolled_model = self._compute_unrolled_model(train_input, train_label, eta, network_optimizer)
		
		val_loss = unrolled_model._loss(val_input, val_label)
		val_loss.backward()

		# compute  dαLval(w',α)
		dalpha = [v.grad for v in unrolled_model._arch_parameters()]

		# compute dw'(Lval(w',α)) # new_w is w'
		dnew_w = [v.grad.data for v in unrolled_model.parameters()]

		# compute formula 8:
		# (dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon) , where w+ = w+dw'Lval(w',α)*epsilon , w- = w-dw'Lval(w',α)*epsilon

		implicit_grads = self._hessian_vector_prodcut(dnew_w, train_input, train_label)
		
		# formula 7: dαLval(w',α)-(dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon)
		for g, ig in zip(dalpha, implicit_grads):
			g.data.sub_(eta, ig.data)

		# update dalpha of self.model
		# note that here alpha is from self.model, dalpha is from unrolled model
		for alpha, dalpha in zip(self.model._arch_parameters(), dalpha):

			if alpha.grad is None:
				alpha.grad = dalpha.data 
			else:
				# ?? why not alpha = alpha - η * dalpha  but directly copy
				# because we just compute the dalpha, upper formula computed by self.optimizer
				alpha.grad.data.copy_(dalpha.data)

	def _compute_unrolled_model(self,train_input, train_label, eta, network_optimizer):

		loss = self.model._loss(train_input, train_label)
		# loss.backward() # compute the gradient
		
		w = concat(self.model.parameters()).data 

		# compute moment: v = old_v * momentum
		try:
			moment = concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul(self.network_momentum)
		except:
			# first time to use moment, init with zeros
			moment = torch.zeros_like(w)
		"""
		res = torch.autograd.grad(loss, self.model.parameters(), allow_unused = True)
		# print(res[0].size())# 741
		
		for (name, _), value in zip(self.model.named_parameters(),res):
			if value is not None:
				logger.info("{}, size:{}".format(name,value.size()))
			else:
				logger.info("{}, None".format(name))
		exit(1)
		"""
		"""
		for name, value in self.model.named_parameters():
			logger.info("name:{}, size:{}".format(name, value.size()))
		"""
		
		dw = concat(torch.autograd.grad(loss, self.model.parameters(), allow_unused=False)).data + w * self.network_weight_decay
		
		new_w = w.sub(eta, moment + dw)

		unrolled_model = self._construct_updated_model(new_w)

		return unrolled_model

	def _construct_updated_model(self, w):

		model_new = self.model.new()
		
		model_dict = self.model.state_dict()

		params, offset = {}, 0
		for name, v in self.model.named_parameters():
			lens = np.prod(v.size())
			# reshape as a tensor
			params[name] = w[offset: offset + lens].view(v.size())
			offset += lens

		assert offset == len(w)
		model_dict.update(params)
		# load param by state_dict
		model_new.load_state_dict(model_dict)
		if self.gpu:
			return model_new.cuda()
		else:
			return model_new

	# compute formula 8:
	# (dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon) , where w+ = w+dw'Lval(w',α)*epsilon , w- = w-dw'Lval(w',α)*epsilon
	def _hessian_vector_prodcut(self, vector, train_input, train_label, r = 1e-2):

		# vector = dw'Lval(w',α)
		# compute epsilon
		eps = r / concat(vector).norm() # norm default is 2-order  res = np.sqrt(sum(pow(item,2))

		# dαLtrain(w+,α)
		for p, v in zip(self.model.parameters(), vector):
			p.data.add_(eps, v)

		loss = self.model._loss(train_input, train_label)
		grad_p = torch.autograd.grad(loss, self.model._arch_parameters())

		# dαLtrain(w-,α) w- = w + dw'Lval(w',α)*epsilon - 2 * dw'Lval(w',α)*epsilon
		for p, v in zip(self.model.parameters(), vector):
			p.data.sub_(2*eps, v)

		loss = self.model._loss(train_input, train_label)
		grad_n = torch.autograd.grad(loss, self.model._arch_parameters())

		# restore the model from w- to w 
		for p, v in zip(self.model.parameters(), vector):
			p.data.add_(eps, v)

		return [(x - y).div_(2 * eps) for x, y in zip(grad_p, grad_n)]



	def _backward_step(self, val_input, val_label):
		loss = self.model._loss(val_input, val_label)
		loss.backward() # compute the gradient with regard to all  


if __name__ == "__main__":

	pass 