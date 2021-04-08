# -*- coding: utf-8 -*-
# @Author: solicucu

import glob 
import os 
import json 
import copy 
import random 
import logging 
import torch 
import torch.nn as nn 
import numpy as np 
from torchvision import datasets, transforms 
from torch.utils.data import Dataset, DataLoader
from PIL import Image 

logger = logging.getLogger("CDNet.data")

data_path = '/home/share/solicucu/data/imagenet/train_val/'

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

		img_path, cid = self.dataset[index]
		img = read_image(img_path)

		if self.transform is not None:
			img = self.transform(img)

		return img, cid

# create DataLoader 
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
	
# batch_size:64 ~20000 batch  need 3 mins to load the data 
def make_data_loader_by_folder(cfg, is_val = False):

	batch_size = cfg.SOLVER.BATCH_SIZE
	num_workers = cfg.DATALOADER.NUM_WORKERS 

	if is_val:
		train_loader = None
	else:
		train_set = datasets.ImageFolder(os.path.join(data_path, 'train'), data_transforms['train'])
		train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = num_workers)

	val_set = datasets.ImageFolder(os.path.join(data_path, 'val'), data_transforms['val'])
	val_loader = DataLoader(val_set, batch_size = batch_size, shuffle = True, num_workers = num_workers)

	
	return train_loader, val_loader, 1000

