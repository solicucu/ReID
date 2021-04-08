# -*- coding: utf-8 -*-
# @Author: solicucu

import torch 
import torch.nn.functional as F 
import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE

root = "./data/"

def load_feats(num_pids):
	tri_file_name = 'tri_feats_20.feats'
	fc_file_name = 'fc_feats_20.feats'
	label_file_name = 'label_pid.txt'
	# tri_file_name = 'bnneck_tri_feats_20.feats'
	# fc_file_name = 'bnneck_fc_feats_20.feats'
	# label_file_name = 'bnneck_label_pid.txt'

	tri_feats = torch.load(root + tri_file_name)
	fc_feats = torch.load(root + fc_file_name)
	labels = []
	with open(root + label_file_name , 'r') as f:
		data = f.readlines()
		for item in data:
			labels.append(item.split(' ')[0])

	print("total sample:", len(labels))

	s  = 0 
	for i in range(len(labels)):
		if labels[i] == str(num_pids):
			s = i 
			break 
	return tri_feats[:s], fc_feats[:s], labels[:s]

def plot_embedding(data, labels, title):
	x_min, x_max = np.min(data, 0), np.max(data, 0)
	data = (data - x_min) / (x_max - x_min)
	# center with 0.5 
	data = data - 0.5 

	plt.xlim(-0.6, 0.6)
	plt.ylim(-0.6, 0.6)

	# plt.subplot(111)
	for x, label in zip(data, labels):
		plt.text(x[0], x[1], str(label), color = plt.cm.Set1(int(label) / 10.),
			fontdict = {'weight': 'bold', 'size':9 })
	plt.xticks([])
	plt.yticks([])
	plt.title(title)
	# move the axis 
	ax = plt.gca()

	ax.xaxis.set_ticks_position('bottom')
	ax.spines['bottom'].set_position(('data',0))

	ax.yaxis.set_ticks_position('left')
	ax.spines['left'].set_position(('data',0))

	ax.spines['top'].set_color('none')  # 设置顶部支柱的颜色为空

	ax.spines['right'].set_color('none')  # 设置右边支柱的颜色为空
	# plt.show()

def visualize_trifeat_and_fc_feat(num_pids = 8):

	tri_feats, fc_feats, labels = load_feats(num_pids)
	# print(tri_feats.size())
	# print(len(labels))
	tri_feats = tri_feats.numpy()
	fc_feats = fc_feats.numpy()
	print('Computing t-SNE embedding')
	tsne = TSNE(n_components=2, init='pca', random_state=0)
	tri_result = tsne.fit_transform(tri_feats)
	fc_result = tsne.fit_transform(fc_feats)
	print("end of computing")
	plt.subplot(1,2,1)
	plot_embedding(tri_result, labels, "distribution of triplet_feats of fblneck")
	plt.subplot(1,2,2)
	plot_embedding(fc_result, labels, 'distribution of softmax_feats of fblneck') 
	plt.show()


def load_fmaps_and_imgs():
	dirs = "D:/project/data/combine/"
	fmap1_name = 'fmap1.fmap' # 64x32
	fmap2_name = 'fmap2.fmap' # 32x16 
	fmap3_name = 'fmap3.fmap' # 16x8
	label_name = 'labels.txt' 

	fmap1 = torch.load(root + fmap1_name)
	fmap2 = torch.load(root + fmap2_name)
	fmap3 = torch.load(root + fmap3_name)
	
	labels = [] 
	with open(root + label_name, 'r') as f:
		data = f.readlines()
		for item in data:
			labels.append(item.strip())

	imgs = []
	for name in labels:
		img_path = dirs + name 
		img = cv2.imread(img_path)
		imgs.append(img) 
		

	batch = len(labels)

	# normalize to (0,1)
	fmap1 = torch.mean(fmap1, dim = 1).numpy()
	fmap2 = torch.mean(fmap2, dim = 1).numpy()
	fmap3 = torch.mean(fmap3, dim = 1).numpy()
	# min number among all 
	for i in range(fmap1.shape[0]):
		min1 = fmap1[i].min()
		max1 = fmap1[i].max()
		fmap1[i] = (fmap1[i] - min1) / (max1 - min1)

	for i in range(fmap2.shape[0]):
		min1 = fmap2[i].min()
		max1 = fmap2[i].max()
		fmap2[i] = (fmap2[i] - min1) / (max1 - min1)

	for i in range(fmap3.shape[0]):
		min1 = fmap3[i].min()
		max1 = fmap3[i].max()
		fmap3[i] = (fmap3[i] - min1) / (max1 - min1)


	fmap1 = np.uint8(np.clip(255 * fmap1, 0, 255))
	fmap2 = np.uint8(np.clip(255 * fmap2, 0, 255))
	fmap3 = np.uint8(np.clip(255 * fmap3, 0, 255))
	
	heatmap1 = []
	heatmap2 = []
	heatmap3 = []
	# w,h
	size = (64, 128)
	for img1,img2,img3 in zip(fmap1, fmap2, fmap3):
		heat1 = cv2.applyColorMap(img1, cv2.COLORMAP_JET)
		heat2 = cv2.applyColorMap(img2, cv2.COLORMAP_JET)
		heat3 = cv2.applyColorMap(img3, cv2.COLORMAP_JET)
		# cv2.resize(size = (w,h))
		heatmap1.append(cv2.resize(heat1, size))
		heatmap2.append(cv2.resize(heat2, size))
		heatmap3.append(cv2.resize(heat3, size))
		
	#img_add = cv2.addWeighted(org_img, 0.3, heat_img, 0.7, 0)
	# 五个参数分别为 图像1 图像1透明度(权重) 图像2 图像2透明度(权重) 叠加后图像亮度
	# cv2.imshow('heapmap3', heatmap1[0])
	# cv2.waitKey(0)
	return [heatmap1, heatmap2, heatmap3] , imgs , labels



def visualize_combine_learning():
	dirs = "D:/project/data/combine/"
	heatmaps, imgs, labels = load_fmaps_and_imgs()
	# add the origin image and heatmap 
	for i in range(len(labels)):
		# for each heatmap 
		for j in range(3):
			img_add = cv2.addWeighted(imgs[i], 0.5, heatmaps[j][i], 0.5, 0)
			# img_add = heatmaps[j][i]
			name = "com{}_".format(j) + labels[i]
			cv2.imwrite(dirs + name, img_add)

	print("end of visualize")


def compare_performance():

	text = ['ShuffleNetv1','MobileNetv2', 'OSNet', 'HA-CNN', 'AutoReID','CNet(ours)',
			'CDNet(our)']
	mAP = [65, 69.5, 81, 75.7, 72.7,  83.5, 83.7]
	r1 = [84.8, 87, 93.6, 91.2, 89.7, 93.6, 93.7]
	param = [1.9, 2.14, 2.2, 2.7, 11.4, 1.44, 1.8]
	# coordinate
	y = [64, 68.5, 80, 74.7, 71.7,  82.1, 83.3]
	x = [1.9, 2.14, 2.2, 2.7, 10.3, 1.24, 2.0]
	plt.subplot(111)
	plt.scatter(param, mAP, c = 'r', marker = '+')
	x_margin = np.arange(1, 12, 1)
	y_margin = np.arange(60, 86, 2.5)

	# add the text
	for i in range(len(text)):
		plt.text(x[i], y[i], text[i], fontdict={'size':9, 'color':'black'}) 

	plt.xticks(x_margin)
	# plt.ylim(60,85)
	plt.yticks(y_margin)
	plt.xlabel("parameters(M)")
	plt.ylabel("mAP(%)")
	plt.show()


if __name__ == "__main__":
	visualize_tirfeat_and_fc_feat()
	# visualize_combine_learning()
	# compare_performance()