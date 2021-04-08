# -*- coding: utf-8 -*-
# @Author: solicucu

import math
import random
import torchvision.transforms as T

class RandomErasing(object):
	"""
	Args:
	probability: the probobility of random erasing
	sl: the minimum of erased_area / image_area
	sh: the maximun of erased_area / image_area
	r: the minmum of aspect ratio of erased region  h / w
	mean: use to replace the pixel value to be erased
	"""
	def __init__(self,probability = 0.5, sl = 0.02, sh = 0.04, r = 0.3, mean=(0.4914, 0.4822, 0.4465)):

		self.prob = probability
		self.sl = sl
		self.sh = sh
		self.r = r 
		self.mean = mean

	def __call__(self, img):

		# if bigger than self.prob,no need to erase
		if random.uniform(0,1) > self.prob:
			return img
		# get the image height and width
		ic, ih, iw = img.size()
		area = ih * iw
		#try 100 times to produce proper region
		for _ in range(100):

			erase_area = random.uniform(self.sl, self.sh)

			#the ratio is range in (0.3, 1/0.3)
			aspect_ratio = random.uniform(self.r, 1./self.r)

			#compute w and h
			h = int(round(math.sqrt(erase_area * aspect_ratio)))
			w = int(round(math.sqrt(erase_area / aspect_ratio)))

			#confirm the both h and w is less than ih and iw respectively
			if h < ih and w < iw :
				# random produe the left coner for the region to be rease
				h1 = random.randint(0, ih - h)
				w1 = random.randint(0, iw - w)

				if ic == 3:
					for i in range(3):
						img[i, h1:h1 + h, w1:w1 + w] = self.mean[i]
				else:
					img[0, h1:h1 + h, w1:w1 + w] = self.mean[0]

				return img
		#if faile to erase image
		return img

# construct transforms for data preprocess
def build_transforms(cfg, is_train = True):

	if is_train:
		transform = T.Compose([
			T.Resize(cfg.DATA.IMAGE_SIZE),
			T.RandomHorizontalFlip(p = cfg.DATA.HF_PROB),
			T.Pad(cfg.DATA.PADDING),
			T.RandomCrop(cfg.DATA.IMAGE_SIZE),
			T.ToTensor(),
			T.Normalize(mean = cfg.DATA.MEAN, std = cfg.DATA.STD),
			RandomErasing(probability = cfg.DATA.RE_PROB)
			])
	else:
		transform = T.Compose([
			T.Resize(cfg.DATA.IMAGE_SIZE),
			T.ToTensor(),
			T.Normalize(mean = cfg.DATA.MEAN, std = cfg.DATA.STD)
			])

	return transform