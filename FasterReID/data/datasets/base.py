# -*- coding: utf-8 -*-
# @Author: solicucu

import numpy as np 

"""
base class of reid dataset
it will mainly compute the number of pids,images and cameras
pids: total person identity
num_imgs: total number of images
num_cams: total number of relavant cameras
it can be used both image or viedo data
"""
class BaseDataset(object):
	# the item in data is 3-tuple (img_path, pid, camid)
	def get_imagedata_info(self, data):
		pids, cams = [], []

		for _, pid, camid in data:

			pids += [pid]
			cams += [camid]
		#unique the elem
		pids = set(pids)
		cams = set(cams)

		num_pid = len(pids)
		num_cams = len(cams)
		num_imgs = len(data)
		return num_pid, num_imgs, num_cams

	#this can be implement for image or video 
	def print_dataset_statistics(self):
		raise NotImplementedError

class BaseImageDataset(BaseDataset):
	"""
	Base class of image reid dataset
	"""

	def print_dataset_statistics(self, train, query, gallery):
		num_train_pids, num_train_imgs, num_train_cams = self.get_imagedata_info(train)
		num_query_pids, num_query_imgs, num_query_cams = self.get_imagedata_info(query)
		num_gallery_pids, num_gallery_imgs, num_gallery_cams = self.get_imagedata_info(gallery)

		print("Dataset statistics:")
		print("  -------------------------------------")
		print("  subset   | # ids | # images | cameras")
		print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
		print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
		print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
