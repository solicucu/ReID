# -*- coding: utf-8 -*-
# @Author: solicucu

import os.path as osp 
from PIL import Image 
from torch.utils.data import Dataset 

#create ImageDataset for dataloader
"""
keep reading the image until succeed
this can avoid IOErro incurred by heavy IO process
"""
def read_image(img_path):
	
	got_img = False
	if not osp.exists(img_path):
		raise IOError("{} doses not exist".format(img_path))
	while not got_img:
		try:
			img = Image.open(img_path).convert("RGB")
			got_img = True
		except IOError:
			print("IOErro incurred when reading {}".format(img_path))

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

		img_path, pid, camid = self.dataset[index]
		img = read_image(img_path)

		if self.transform is not None:
			img = self.transform(img)

		return img, pid, camid, img_path

