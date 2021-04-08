# -*- coding: utf-8 -*-
# @Author: solicucu

import copy 
import random
import torch
import numpy as np 

from collections import defaultdict
from torch.utils.data.sampler import Sampler 


class TripletSampler(Sampler):
	"""
	randomly sample N identities,for each identity randomly sample K instances
	thus, batch size is N * K 
	Args:
	data (list): item is (img_path, pid, camid)
	num_instances (int): number of instances per identity
	batch_size (int): number of examples in a batch

	"""
	def __init__(self, data, batch_size, num_instances):

		self.data = data
		self.batch_size = batch_size
		# K
		self.num_instances = num_instances
		# compute the N
		self.num_pids_per_batch = self.batch_size // self.num_instances
		# use pid as key, value is all index of same pid
		# default value is list
		self.index_dict = defaultdict(list)
		for index, (_, pid, _) in enumerate(self.data):
			self.index_dict[pid].append(index)
		self.pids = list(self.index_dict.keys())
		# estimate number of examples in an epoch
		self.length = 0
		for pid in self.pids:
			idxs = self.index_dict[pid]
			num = len(idxs)
			#insufficient pid will repeated to satify num_instances
			if num < self.num_instances:
				num = self.num_instances
			#remainders are ignored
			self.length += (num - num % self.num_instances)
	# return a iterable object
	def __iter__(self):

		# key is pid, value is a list, elem is list contains self.num_instances
		batch_idxs_dict = defaultdict(list)

		for pid in self.pids:
			idxs = copy.deepcopy(self.index_dict[pid])

			if len(idxs) < self.num_instances:
				#select num_instances which can be repeated
				idxs = np.random.choice(idxs,size = self.num_instances, replace = True)
			# be careful for the indention	!!!!!!!!!!!
			random.shuffle(idxs)
			#continuously select num_instances as a combination
			batch_idxs = []
			for idx in idxs:
				batch_idxs.append(idx)
				if len(batch_idxs) == self.num_instances:
					batch_idxs_dict[pid].append(batch_idxs)
					batch_idxs = []



		#copy a temp pid
		temp_pids = copy.deepcopy(self.pids)
		final_idxs = []
		
		while len(temp_pids) >= self.num_pids_per_batch:

			# randomly select n pids
			selected_pids = random.sample(temp_pids, self.num_pids_per_batch)

			for pid in selected_pids:

				batch_idxs =  batch_idxs_dict[pid].pop(0)
				final_idxs.extend(batch_idxs)

				if len(batch_idxs_dict[pid]) == 0:
					temp_pids.remove(pid)

		self.length = len(final_idxs)

		return iter(final_idxs)

	def __len__(self):

		return self.length
 			
 				