# -*- coding: utf-8 -*-
# @Author: solicucu

class AverageMeter(object):

	def __init__(self):

		self.reset()

	def reset(self):

		self.avg = 0.
		self.sum = 0.
		self.count = 0.

	def update(self, val, n = 1):

		self.sum += val * n
		self.count += n 
		self.avg = self.sum / self.count

class RunningAverageMeter(object):

	def __init__(self, alpha = 0.98):

		self.reset()
		self.alpha = alpha 

	def reset(self):

		self.avg = 0.

	def update(self, val):

		if self.avg == 0. :
			self.avg = val 
		else:
			self.avg = self.avg * self.alpha + (1 - self.alpha) * val
			
