# -*- coding: utf-8 -*-
# @Author: solicucu

import numpy as np  
import logging 
import torchvision.transforms as T 
from torch.utils.data import DataLoader
from .transforms import RandomErasing 
from .datasets import init_dataset, ImageDataset
from .samplers import TripletSampler
from .collate_batch import train_collate_fn ,val_collate_fn 
from collections import defaultdict 

# construct transforms for data preprocess
def build_transforms(is_train = False):

	if is_train:
		transform = T.Compose([
			T.Resize((256,128)),
			T.RandomHorizontalFlip(p = 0.5),
			T.Pad(10),
			T.RandomCrop((256,128)),
			T.ToTensor(),
			T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
			RandomErasing(probability = 0.5)
			])
	else:
		transform = T.Compose([
			T.Resize((256,128)),
			T.ToTensor(),
			T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
			])

	return transform
"""
split the dataset.train into train and val for training
train: use to update the network weights
val: use to update the architecture weights (alpha)
the format of item in data: (imgpath, pid, camid)

Retruns:
train_set: divisible for num_instance for each pid
val: each pid has num_instance 
"""
def split_train_and_val(data, num_instance):

	# print(len(data)) # 12936
	# compute the number instances for each identity
	pid_dicts = defaultdict(list)
	

	for item in data:
		_, pid, _ = item
		pid_dicts[pid].append(item)
	"""
	split the data into train and val
	random choose 4 instance for each pid 
	do not delete if the number of idenity less than 4
	if number > 4, rm those beyond  4 times
	"""
	def get_list_by_index(data, index):

		select = []
		for i in index:
			select.append(data[i])
		return select

	train_set, val_set = [], []
	for key, item in pid_dicts.items():
		num = len(item)
		# random.choice just use for 1-dimension list, so we work on index
		index = range(num)
		if num < num_instance:
			select = np.random.choice(index, size = num_instance, replace = True)
			val_set.extend(get_list_by_index(item, select))

			select = np.random.choice(index, size = num_instance, replace = True)
			train_set.extend(get_list_by_index(item, select))

		else:
			num_rm = num % num_instance
			if num_rm == 0:
				num_rm = num_instance
			index_rm = np.random.choice(index, size = num_rm, replace = False)
			val_set.extend(get_list_by_index(item, index_rm))
			# two type to rm the item 
			if num_rm != num_instance or (num_rm == num_instance and num >= 2 * num_instance):
				# delete must from end to begin, if not so, the index will change
				index_rm = sorted(index_rm, reverse = True)
				# print(index_rm)
				for rm in index_rm:
					item.pop(rm)

			rest_size = num_instance - num_rm
			index = range(len(item)) # update the index 
			rest_index = np.random.choice(index, size = rest_size, replace = False)

			val_set.extend(get_list_by_index(item, rest_index))
			train_set.extend(item)

	return train_set, val_set

	"""
	# construct num_pid
	num_pid = {} # key is number instance, value is the number of pid 
	for key in pid_dicts:
		num = len(pid_dicts[key])
		if num in num_pid:
			num_pid[num] += 1
		else:
			num_pid[num] = 1
	"""
	# show the result 
	# sorted by key default
	# for key in sorted(num_pid):
	# 	print(key, num_pid[key])
	""" less than 8 
	2 3
	3 12
	4 12
	5 31
	6 24
	7 40
	8 39
	"""

def make_data_loader(cfg):
	# CSNet_Search must be same can be get, name after by . can be any str
	logger = logging.getLogger("CDNet_Search.data")
	train_transforms = build_transforms(is_train = True)
	val_transforms = build_transforms()

	num_workers = cfg.DATALOADER.NUM_WORKERS
	batch_size = cfg.SOLVER.BATCH_SIZE
	num_instance = cfg.DATALOADER.NUM_INSTANCE

	# init the dataset
	# has self.train self.query, self.gallery with item (imgpath, pid, camid)
	dataset = init_dataset(cfg.DATA.DATASET, cfg.DATA.DATASET_DIR)

	num_classes = dataset.num_train_pids 
	# print(num_classes) 751

	train, val = split_train_and_val(dataset.train, num_instance)
	logger.info("size of train_set is {}".format(len(train)))
	logger.info("size of val_set is {}".format(len(val)))

	# 11160 + 3004
	train_set = ImageDataset(train, train_transforms)
	val_set = ImageDataset(val, train_transforms)

	# create dataloader
	if cfg.DATALOADER.SAMPLER == 'triplet':
		train_loader = DataLoader(train_set,
			batch_size = batch_size,
			sampler = TripletSampler(train, batch_size, num_instance),
			num_workers = num_workers,
			collate_fn = train_collate_fn)
		val_loader = DataLoader(val_set,
			batch_size = batch_size,
			sampler = TripletSampler(val, batch_size, num_instance),
			num_workers = num_workers,
			collate_fn = train_collate_fn)
	else:
		raise RuntimeError("{} not know sampler".format(cfg.DATALOADER.SAMPLER))

	test_set = ImageDataset(dataset.query + dataset.gallery,val_transforms)
	test_loader = DataLoader(
		test_set,
		shuffle = False,
		batch_size = batch_size,
		num_workers = num_workers,
		collate_fn = val_collate_fn
		)

	return train_loader, val_loader, test_loader, len(dataset.query), num_classes