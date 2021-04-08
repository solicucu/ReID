# -*- coding: utf-8 -*-
# @Author: solicucu

import numpy as np 

def eval_func(dist_mat, q_pids, q_camids, g_pids, g_camids, max_rank = 50):

	num_q, num_g = dist_mat.shape

	if num_g < max_rank:

		max_rank = num_g
		print("Note: number of gallery samples is small than max_rank , got {}".format(num_g))

	# sort the dist_mat by distance in dimension 1
	# return the index sorted
	indices = np.argsort(dist_mat, axis =1)
	"""
	here,
	indices: num_q x num_g
	g_pids: 1xnum_g
	g_pids[indices]-> num_q x num_g, equals to for each row in indice, produce the g_pids row according the sorted indexs
	q_pids: 1x num_q
	q_pids[:,np.newaxis]: num_q x 1
	g_pids[indices] == q_pids[:, np.newaxis] : num_q x num_g 
	 -> equals to for each row in q_pids,determine pids == the same row in g_pids ?
	"""
	matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

	# compute cmc curve for each query 
	all_cmc = []
	all_AP = []
	# number of valid query except the same camids 
	num_valid_q = 0.
	for q_idx in range(num_q):

		q_pid = q_pids[q_idx]
		q_camid = q_camids[q_idx]

		#remove the gallery samples that have sampe pids and camids with the query

		order = indices[q_idx]
		remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)

		keep = np.invert(remove)

		# compute cmc curve
		# a binary vector, position with True denotes correct mathes
		orig_cmc = matches[q_idx][keep]
		res = np.sum(orig_cmc, axis = -1)

		if not np.any(orig_cmc):
			# this condition is true when query identity does not appear in the gallery
			continue
		cmc = orig_cmc.cumsum()
		cmc[cmc>1] = 1
		all_cmc.append(cmc[:max_rank])
		num_valid_q += 1

		# compute average percision
		# get the number of same identity with query in gallery
		num_rel = orig_cmc.sum()
		tmp_cmc = orig_cmc.cumsum()
		# compute the correctly matches in k retrieval result propotion
		# just like precision
		tmp_cmc = [ x / (i + 1.) for i, x in enumerate(tmp_cmc)]
		# just consider the result of correctly matches at 1,2,3,...
		tmp_cmc = np.asarray(tmp_cmc) * orig_cmc

		AP = tmp_cmc.sum() / num_rel
		all_AP.append(AP)

	assert num_valid_q > 0 , "Error: all query identity do not appear in the gallery"

	# compute all query
	all_cmc = np.asarray(all_cmc).astype(np.float32) 
	all_cmc = all_cmc.sum(0) / num_valid_q
	mAP = np.mean(all_AP)

	return all_cmc, mAP


