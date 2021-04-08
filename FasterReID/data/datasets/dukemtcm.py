# -*- coding: utf-8 -*-
# @Author: solicucu

import os 
import glob 
import re 
import urllib
import zipfile 
import os.path as osp 
from .base import BaseImageDataset


class DukeMTMC(BaseImageDataset):
	# Dataset statistics:
	# identities: 1404 (train + query)
	# images:16522 (train) + 2228 (query) + 17661 (gallery)
	# cameras: 8

	def __init__(self, root = '', verbose = True, **kwargs):

		super(DukeMTMC, self).__init__()
		self.dataset_dir = root + "dukemtmc/"
		self.data_url = "http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-reID.zip"
		self.train_dir = self.dataset_dir + "bounding_box_train/"
		self.query_dir = self.dataset_dir + "query/"
		self.gallery_dir = self.dataset_dir + "bounding_box_test/"

		self._download_data()
		self._check_before_run()

		self.train = self._process_dir(self.train_dir, relabel = True)
		self.query = self._process_dir(self.query_dir)
		self.gallery = self._process_dir(self.gallery_dir)

		if verbose:
			print("=> DukeMTMC dataset is loaded")
			self.print_dataset_statistics(self.train, self.query, self.gallery)

		self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
		self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
		self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)


	def _check_before_run(self):

		if not osp.exists(self.dataset_dir):
			raise RuntimeError("{} is not available".format(self.dataset_dir))
		if not osp.exists(self.train_dir):
			raise RuntimeError("{} is not available".format(self.train_dir))
		if not osp.exists(self.query_dir):
			raise RuntimeError("{} is not available".format(self.query_dir))
		if not osp.exists(self.gallery_dir):
			raise RuntimeError("{} is not available".format(self.gallery_dir))

	def _download_data(self):

		if osp.exists(self.dataset_dir):
			print("the dataset has been downloaded")
			return 
		print("Create directory {}".format(self.dataset_dir))

		os.makedirs(self.dataset_dir)
		fpath = osp.join(self.dataset_dir, osp.basename(self.data_url))
		print("Download dukemtmc dataset")
		urllib.request.urlretrieve(self.data_url, fpath)

		print("Extracting files")
		zip_ref = zipfile.ZipFile(fpath, 'r')
		zip_ref = extractall(self.dataset_dir)
		zip_ref.close()

	def _process_dir(self, path, relabel = False):

		img_paths = glob.glob(osp.join(path, '*.jpg'))

		pattern = re.compile(r'([-\d]+)_c(\d)')

		pids = set()
		for path in img_paths:
			# change the pid and camid to int , ignore camid
			pid, _ = map(int, pattern.search(path).groups())
			pids.add(pid)

		# construct a dict with pid as key, vlaue is label
		pid2label = {pid: label for label, pid in enumerate(pids)}

		dataset = []
		for path in img_paths:
			pid, camid = map(int, pattern.search(path).groups())
			assert 1<= camid <= 8 
			camid -= 1 # index start from 0 
			if relabel:
				pid = pid2label[pid]

			dataset.append((path, pid, camid))

		return dataset

if __name__ == "__main__":

	data = DukeMTMC(root = "D:/project/data/")