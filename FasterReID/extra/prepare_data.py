# -*- coding: utf-8 -*-
# @Author: solicucu

import os 
import glob 
import torchvision.transforms as T
from PIL import Image 
from collections import defaultdict 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# root = 'D:/project/data/market1501/bounding_box_train/'
root = '/home/share/solicucu/data/market1501/bounding_box_train/'

# return n identies (pid, label, path)
def get_data(n, min_num = 20):
	names = os.listdir(root) 
	pid2list = defaultdict(list)
	for name in names: 
		# print(name)
		if name == "Thumbs.db":
			continue
		pid = int(name.split('_')[0])

		path = root + name 
		pid2list[pid].append(path) 


	pids = list(pid2list.keys())
	# num = len(pids)
	data = [] 
	count = 0 
	for pid in pids:
		# constaint the ids number >= min_num 
		if len(pid2list[pid]) < min_num:
			continue
		item = [(count,path,pid) for path in pid2list[pid]]
		data.extend(item)

		count += 1
		if count == n: 
			break

	return data 

def get_all_data():

	path = '/home/share/solicucu/data/combine/'
	names = os.listdir(path)
	data = [] 
	for name in names:
		pid = int(name.split('_')[0])
		img_path = path + name 
		data.append((name, img_path, pid))

	return data 

val_transform = T.Compose([
			T.Resize([256,128]),
			T.ToTensor(),
			T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
			])	

def read_image(img_path):
	
	got_img = False
	if not os.path.exists(img_path):
		raise IOError("{} doses not exist".format(img_path))
	while not got_img:
		try:
			img = Image.open(img_path).convert("RGB")
			got_img = True
		except IOError:
			print("IOErro incurred when reading {}".format(img_path))

	return img 

class ImageDataset(Dataset):

	def __init__(self, dataset, transform = None):

		self.dataset = dataset 
		self.transform = transform 

	def __len__(self):

		return len(self.dataset)

	def __getitem__(self, index): 

		label, path, pid = self.dataset[index]
		img = read_image(path)

		if self.transform is not None: 
			img = self.transform(img) 

		return img, label, pid 

# return a dataloader and number_class 
def make_dataloader(num_pids=5, batch_size = 64, num_workers = 4):

	data = get_data(num_pids)
	print("select images:", len(data))
	dataset = ImageDataset(data, transform = val_transform)
	data_loader = DataLoader(dataset, batch_size= batch_size, shuffle = False, num_workers = num_workers)
	# 751 for market1501
	return data_loader, 751 

def make_combine_dataloader():

	data = get_all_data()
	print("select images:", len(data))
	dataset = ImageDataset(data, transform = val_transform)
	data_loader = DataLoader(dataset, batch_size = 64, shuffle = False, num_workers = 1)
	# 751 for market1501
	return data_loader, 751 

if __name__ == '__main__':
	data = get_data(2)

	for item in data:
		print(item)
		