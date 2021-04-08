# -*- coding: utf-8 -*-
# @Author: solicucu

import os 
import torch
from PIL import Image 
from torch.utils.data import DataLoader
from .transforms import build_transforms
from .datasets import init_dataset, ImageDataset 
from .samplers import TripletSampler 
from .collate_batch import train_collate_fn, val_collate_fn 


def make_data_loader(cfg):
	train_transforms = build_transforms(cfg, is_train = True)
	val_transfroms = build_transforms(cfg, is_train = False)

	num_workers = cfg.DATALOADER.NUM_WORKERS

	# init the dataset
	# get the train, val, test data in the form of list  respectively
	dataset = init_dataset(cfg.DATA.DATASET, cfg.DATA.DATASET_DIR)


	num_classes = dataset.num_train_pids
	# create ImageDataset 
	# it will store the data info list such as (img_path, pid, camid) and 
	# return 4-tuple(img, pid, camid, img_path) via given index (from sampler)
	train_set = ImageDataset(dataset.train, train_transforms)

	# create dataloader
	# 11776 samplers # sampler is used to produce the index 
	if cfg.DATALOADER.SAMPLER == 'triplet':
		train_loader = DataLoader(
			train_set, batch_size = cfg.SOLVER.BATCH_SIZE,
			sampler = TripletSampler(dataset.train, cfg.SOLVER.BATCH_SIZE, cfg.DATALOADER.NUM_INSTANCE),
			num_workers = num_workers, collate_fn = train_collate_fn
			)
	else:
		raise RuntimeError("{} not know sampler".format(cfg.DATALOADER.SAMPLER))

	
	val_set = ImageDataset(dataset.query + dataset.gallery, val_transfroms)

	val_loader = DataLoader(
			val_set, batch_size = cfg.TEST.IMGS_PER_BATCH, shuffle = False,
			num_workers = num_workers, collate_fn = val_collate_fn
		)

	return train_loader, val_loader, len(dataset.query), num_classes

import re
"""
Args:
	cfg: config params
	img_path: a list contains image absolute path
return:
	return a list of data with elem-list [imgs, pids, camids, img_paths]
"""
def make_batch_data(cfg, img_paths):
	imgs = []
	pids = []
	camids = []
	# 图片预处理
	val_transfroms = build_transforms(cfg, is_train = False)
	for path in img_paths:
		# 判断图片是否存在
		if not os.path.exists(path):
			raise IOError("{} doses not exist".format(path))

		img = Image.open(path).convert("RGB")
		img = val_transfroms(img)
		# 获取图片的id 和 camid
		pattern = re.compile(r'([-\d]+)_c(\d)')
		res = pattern.search(path).groups()
		pid, camid = map(int, res)
		img = img.unsqueeze(0)
		imgs.append(img)
		pids.append(pid)
		camids.append(camid)

	if len(imgs) > 1:
		imgs = torch.cat(imgs, dim = 0)

	return [imgs, pids, camids, img_paths]



if __name__ == "__main__":

	# image = torch.randn(3,384,128)
	"""
	image = get_image("1.jpg")
	train_transform = build_transforms(cfg, is_train = True)

	img = train_transform(image[0])

	print(img.size())
	"""
	# make_data_loader(cfg)
	pass 