# -*- coding: utf-8 -*-
# @Author: solicucu

import glob
import re 
import os.path as osp  
from .base import BaseImageDataset

"""
Market1501
Reference:
Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
URL: http://www.liangzheng.org/Project/project_reid.html

Dataset statistics:
# identities: 1501 (+1 for background)
# images: 12936 (train) + 3368 (query) + 15913 (gallery)
"""
class Market1501(BaseImageDataset):

	dataset_name = 'market1501'

	def __init__(self, root = '', verbose = True):

		super(Market1501, self).__init__()
		self.dataset_dir = osp.join(root, self.dataset_name)
		self.train_dir = osp.join(self.dataset_dir,'bounding_box_train')
		self.query_dir = osp.join(self.dataset_dir,'query')
		self.gallery_dir = osp.join(self.dataset_dir,'bounding_box_test')
		#check the directory whether valid or not
		self.check_before_run()

		self.train = self.process_dir(self.train_dir, relabel = True)
		self.query = self.process_dir(self.query_dir)
		self.gallery = self.process_dir(self.gallery_dir)

		if verbose:
			print("=> init Market1501")
			self.print_dataset_statistics(self.train, self.query, self.gallery)

		self.num_train_pids, self.num_train_imgs, self.num_train_camids = self.get_imagedata_info(self.train)
		self.num_query_pids, self.num_query_imgs, self.num_query_camids = self.get_imagedata_info(self.query)
		self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_camids = self.get_imagedata_info(self.gallery)
	
	def check_before_run(self):

		if not osp.exists(self.dataset_dir):
			raise RuntimeError("{} is not available".format(self.dataset_dir))
		if not osp.exists(self.query_dir):
			raise RuntimeError("{} is not available".format(self.query_dir))
		if not osp.exists(self.train_dir):
			raise RuntimeError("{} is not available".format(self.train_dir))
		if not osp.exists(self.gallery_dir):
			raise RuntimeError("{} is not available".format(self.gallery_dir))
	
	# relable: relable for the pids , because they are not continuous
	def process_dir(slef, dir_path, relabel = False):
		# list all image in the specified path
		img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
		# first group is pid(note that some pid begin with - ), second group is camid
		pattern = re.compile(r'([-\d]+)_c(\d)')
		# used to store all pids
		pid_sets = set()
		for img_path in img_paths:
			# map the pid(str type) to int (int type)  
			res = pattern.search(img_path).groups()
			pid, _ = map(int, res)
			
			# pid == -1 denote the junk images, we just ignored
			if pid == -1:
				continue 
			pid_sets.add(pid)
		# construct a dict pid2lable
		pid2label = {pid: label for label, pid in enumerate(pid_sets)}


		dataset = []
		for img_path in img_paths:

			pid, camid = map(int, pattern.search(img_path).groups())
			if pid == -1:
				continue
			assert 0<= pid <= 1501
			assert 1<= camid <= 6
			#change camid index start from 0
			camid -=1 
			if relabel:
				pid = pid2label[pid]
			dataset.append((img_path, pid, camid))

		return dataset


if __name__ == "__main__":

	dataset = Market1501("D:\\project\\data\\")
	
	print(dataset.num_train_pids)
	print(dataset.num_train_imgs)
	print(dataset.num_train_camids)