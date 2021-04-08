# -*- coding: utf-8 -*-
# @Author: solicucu

import torch
import numpy as np 

from .reid_val import eval_func 

class R1_mAP(object):

	def __init__(self, num_query, max_rank = 50, use_gpu = False, feat_norm = 'yes'):

		self.num_query = num_query
		self.max_rank = max_rank
		self.feat_norm = feat_norm
		self.use_gpu = use_gpu
		self.reset()

	def reset(self):

		self.feats = []
		self.pids = []
		self.camids = []
		self.paths = []
	#batch contains feat,pid,camid
	def update(self, batch):

		feat, pid, camid = batch 
		self.feats.append(feat)
		self.pids.extend(np.asarray(pid))
		self.camids.extend(np.asarray(camid))
		# self.paths.extend(np.asrray(path))

	def compute(self):

		feats = torch.cat(self.feats, dim = 0)
		if self.feat_norm == 'yes':
			print("the test feature is normalized")
			feats = torch.nn.functional.normalize(feats, dim = 1, p = 2)
		# query
		qf = feats[:self.num_query]
		q_pids = np.asarray(self.pids[:self.num_query])
		q_camids = np.asarray(self.camids[:self.num_query])

		# gallery
		gf = feats[self.num_query:]
		g_pids = np.asarray(self.pids[self.num_query:])
		g_camids = np.asarray(self.camids[self.num_query:])

		m,n = qf.shape[0], gf.shape[0]
		# compute the euclidian distance

		dist_mat = torch.pow(qf, 2).sum(dim = 1, keepdim = True).expand(m, n) + \
		           torch.pow(gf, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
		dist_mat.addmm_(1, -2, qf, gf.t())

		if self.use_gpu:
			dist_mat = dist_mat.cpu().numpy()
			
		else:
			dist_mat = dist_mat.numpy()

		cmc, mAP = eval_func(dist_mat, q_pids, q_camids, g_pids, g_camids)

		return cmc, mAP