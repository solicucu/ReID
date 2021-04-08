# -*- coding: utf-8 -*-
# @Author: solicucu

import os 
import numpy as np 
import glob 
import json 
import copy 
import math
import random 
import logging 
import torch  
from torchvision import datasets, transforms 
from torch.utils.data import Dataset, DataLoader 
from PIL import Image 

logger = logging.getLogger("CDNet.data")

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

class CUTOUT(object):

	def __init__(self, length):
		self.length = length

	def __call__(self, img):
		h, w = img.size(1), img.size(2)
		mask = np.ones((h,w), np.float32)
		# produce the center point to be cutout
		y = np.random.randint(h)
		x = np.random.randint(w)
		# get the upper left and lower right corner
		y1 = np.clip(y - self.length // 2, 0, h) 
		y2 = np.clip(y + self.length // 2, 0, h)
		x1 = np.clip(x - self.length // 2, 0, w) 
		x2 = np.clip(x + self.length // 2, 0, w)

		mask[y1:y2, x1:x2] = 0. 
		mask = torch.from_numpy(mask) 
		# expand as img shape
		mask = mask.expand_as(img) 
		img *= mask 
		return img  

def get_transforms(name, types = 'train', cutout = 0, size = [32, 32]):

	if name == 'cifar10':
		mean = [x / 255 for x in [125.3, 123.0, 113.9]]
		std = [x / 255 for x in [63.0, 62.1, 66.7]]

	elif name == 'cifar100' or name == 'cifar100_combine':
		mean = [x / 255 for x in [129.3, 124.1, 112.4]]
		std = [x / 255 for x in [68.2, 65.4, 70.4]]

	else:
		raise TypeError("unkonw dataset: {}".format(name))

	# data Augmentation
	lists = [
			transforms.Resize(size),
			transforms.RandomHorizontalFlip(p = 0.5),
			transforms.Pad(4),
			# transforms.RandomCrop(64, padding=4), 
			transforms.RandomCrop(size),
			transforms.ToTensor(), 
			transforms.Normalize(mean, std)
	]
	if cutout > 0:
		lists.extend([CUTOUT(cutout)]) 
		# lists.extend([RandomErasing()])

	data_transforms = {
		'train': transforms.Compose(lists),
		'test': transforms.Compose([
			transforms.Resize(size),
			transforms.ToTensor(), 
			transforms.Normalize(mean, std)
		])
	}

	return data_transforms[types]


# create Dataset
#create ImageDataset for dataloader
"""
keep reading the image until succeed
this can avoid IOErro incurred by heavy IO process
"""
def read_image(img_path):
	
	got_img = False
	if not os.path.exists(img_path):
		raise IOError("{} doses not exist".format(img_path))
	while not got_img:
		try:
			img = Image.open(img_path).convert("RGB")
			got_img = True
		except IOError:
			logger.info("IOErro incurred when reading {}".format(img_path))

	return img 

"""
need to rewrite __init__, __len__, __getitem__
"""
class ImageDataset(Dataset):

	def __init__(self, dataset, transform = None):

		self.dataset = dataset
		self.transform = transform

	def __len__(self):

		return len(self.dataset)

	def __getitem__(self, index):

		# img_path, cid = self.dataset[index]
		# img = read_image(img_path)
		# follow is for the img is PIL format 
		img, cid = self.dataset[index]

		if self.transform is not None:
			img = self.transform(img)

		return img, cid

def make_data_loader_by_folder(cfg, is_val = False):

	
	batch_size = cfg.SOLVER.BATCH_SIZE
	num_workers = cfg.DATALOADER.NUM_WORKERS

	name = cfg.DATA.DATASET
	cutout = cfg.DATA.CUTOUT 
	size = cfg.DATA.IMAGE_SIZE
	# name only in [cifar10, cifar100]
	assert name in ['cifar10', 'cifar100', 'cifar100_combine'], "not know dataset {}".format(name)

	data_path = cfg.DATA.DATASET_DIR + "{}/".format(name)
	train_transform = get_transforms(name, "train", cutout, size = size)
	test_transform = get_transforms(name, 'test', size = size)
	
	if is_val:
		train_loader = None 
	else:
		train_set = datasets.ImageFolder(os.path.join(data_path, 'train'), train_transform )
		train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = num_workers)
	
	test_set = datasets.ImageFolder(os.path.join(data_path, 'test'), test_transform)
	test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = True, num_workers = num_workers)

	num_class = 10 if name == 'cifar10' else 100 

	return train_loader, test_loader, num_class 


"""
this function is used to create cifar100_combine from cifar100
each subclass directory is moved as first order directory
"""
import shutil 
def combine_cifar100(origin_path, target_path):
	names = os.listdir(origin_path)
	for name in names:
		class_names = os.listdir(origin_path + name)
		for class_name in class_names:
			cur_dir = origin_path + "{}/{}/".format(name, class_name)
			new_class_dir = target_path + "{}_{}/".format(name,class_name)
			if not os.path.exists(new_class_dir):
				os.makedirs(new_class_dir)
			files = os.listdir(cur_dir)
			for file in files:
				src_file = cur_dir + file 
				target_file = new_class_dir + file  
				shutil.copyfile(src_file, target_file)
	print("end of copying")



if __name__ == "__main__":
	dataset_path = "/home/share/solicucu/data/CIFAR/"
	# dir2cid, _ = dirs2cids(dataset_path, dataset = 'cifar100', save = True)
	# split_train_val(dataset_path + "cifar10/")
	# combine_cifar100(dataset_path + 'cifar100/test/', dataset_path + 'cifar100_combine/test/')