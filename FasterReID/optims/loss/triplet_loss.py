# -*- coding: utf-8 -*-
# @Author: solicucu


import torch 
import torch.nn as nn
# -1 denotes compute the lastest dimension
def normalize(x, axis = -1):

	# normalize x to unit length along the specified dimension
	# len = (sum(xi^k)) -> 开k次根号，here k is 2
	x_len = torch.norm(x, 2, axis, keepdim = True) 

	x = 1. * x / (x_len.expand_as(x) + 1e-12)

	return x 

"""
# compute the euclidean for each identity i to any other j
# the elem of matri dist(i,j) is distance from i to j
# given m identities and another n identities, the return matrix is m x n 
Args:
	x: pytorch tensor, with shape [m, d]
	y: pytroch tensor, with spape [n, d]

Returns:
	dist_matix: pytorch tensor, with shape [m, n]

compute formula:
(x-y)^2 = x^2 + y^2 - 2xy
mind that x is a vector

"""
def euclidean_dist(x,y):

	m, n = x.size(0), y.size(0)
	# compute x^2, y^2
	# expand the same col to compute with y
	xx = torch.pow(x,2).sum(1, keepdim = True).expand(m, n)

	yy = torch.pow(x,2).sum(1, keepdim = True).expand(n, m).t()

	dist = xx + yy 

	# compute x * y -> res, and then, compute (dist + (-2 * res ))
	dist.addmm_(1, -2, x, y.t())
	# for numerical stability
	dist = dist.clamp(min = 1e-12).sqrt()

	return dist

"""
Args:
	dist_mat: pytorch tensor, with shape [N,N]dist_mat(i,j) denotes the distance between i and j 
	labels: pytorch LongTensor, with shape [N] denotes ids
	return_inds: whether to return the indices of ap and an

Returns:
	dist_ap: the distance between anchor and hardest positive sample
	dist_an: the distance between anchor and hardest negative sample
	p_ind, n_ind : the relative index for hardest positive and negative sample
"""
def hard_example_mining(dist_mat, labels, return_inds = False):

	assert len(dist_mat.size()) == 2
	assert dist_mat.size(0) == dist_mat.size(1)

	N = dist_mat.size(0)

	# find the same identity or not 
	# with result shape [N, N] -> bool
	is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
	is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

	# extact pos example
	# to correctly use view(N, -1), it must promise that for each identiy,
	# they need to have the same number of positive number, if not , it will go wrong
	pos = dist_mat[is_pos].contiguous().view(N, -1)
	# select the most disimilar positive identity
	# with dist_ap means dist(anchor, hard_positive)
	# both dist_ap, p_inds with shape [N, 1]
	dist_ap, p_inds = torch.max(pos, 1, keepdim = True)

	neg = dist_mat[is_neg].contiguous().view(N, -1)
	#select the most similar negative identity
	dist_an, n_inds = torch.min(neg, 1, keepdim = True)

	# squeeze 1 channel
	dist_ap, dist_an = dist_ap.squeeze(1), dist_an.squeeze(1)

	if return_inds:
		# shape [N, N]
		# contruct a matrix
		# for each row is range(0,n)
		ind = (labels.new()).resize_as_(labels) \
				.copy_(torch.arange(0, N).long()) \
				.unsqueeze(0).expand(N, N)
		p_ind = torch.gather(ind[is_pos].contiguous().view(N, -1), 1, p_inds.data )
		n_ind = torch.gather(ind[is_neg].contiguous().view(N, -1), 1, n_inds.data )

		return dist_ap, dist_an, p_ind, n_ind 

	return dist_ap, dist_an


"""
for each anchor, find the hardest positive and negative sample

Args:
	margin: the margin for triplet loss
Returns:
	loss: triplet loss for given dist_ap, dist_an
	dist_ap: the distance between anchor and hardest positive sample
	dist_an: the distance between anchor and hardest negative sample

"""


class TripletLoss(object):

	def __init__(self, margin = 0):

		self.margin = margin

		if margin != 0 :

			self.rank_loss = nn.MarginRankingLoss(margin = self.margin)
		else:
			# do not  specify the margin, so it will continously make dist_an  >> dist_ap
			self.rank_loss = nn.SoftMarginLoss()

	def __call__(self, global_feat, labels, normalize_feat = False):

		# normalize the feature vector
		if normalize_feat:

			global_feat = normalize(global_feat, axis = -1)

		#compute the dist_mat for hard example mining

		dist_mat = euclidean_dist(global_feat, global_feat)

		#for given anchor identity, select the most similar negtive example and most disimilar positive example
		dist_ap, dist_an = hard_example_mining(dist_mat, labels)

		# ranking_loss,nn.MarginRankingLoss(x1,x2,y)
		# -> given y , if y == 1, expect x1 is much larger while x2 is much smaller
		y = dist_an.new().resize_as_(dist_an).fill_(1)	

		if self.margin != 0:

			loss = self.rank_loss(dist_an,dist_ap, y)
		else:

			loss = self.rank_loss(dist_an-dist_ap, y)

		return loss, dist_ap, dist_an



		
class CrossEntropyLabelSmooth(nn.Module):
	"""
	crossentopy:
	loss = sum(-qi * log(pi) for i in range(batch_size)) 
	where qi = 1 if i == labeli else 0

	label smooth:
	now change the q more smooth 
	qi = (1- epsilon) * qi + epsilon / num_class
	where qi = 1 if i == labeli else 0 for later qi.

	"""

	def __init__(self, num_classes, epsilon = 0.1, use_gpu = False):

		super(CrossEntropyLabelSmooth, self).__init__()
		self.num_classes = num_classes
		self.epsilon = epsilon
		self.use_gpu = use_gpu
		self.logsoftmax = nn.LogSoftmax(dim = 1)

	def forward(self, scores, labels):

		"""
		args:
			scores: predict matrix (before softmax) with shape [batch_size, num_classes]
			labels: ground true labels with shape [num_classes]
		"""
		# compute log(pi)
		log_probs = self.logsoftmax(scores)
		q = torch.zeros(log_probs.size()).scatter_(dim = 1, index = labels.unsqueeze(1).data.cpu(), value = 1)
		# since q is a new tensor
		if self.use_gpu:
			q = q.cuda()
		# smooth the q 
		q = (1 - self.epsilon) * q  + self.epsilon / self.num_classes

		loss = (-q * log_probs).mean(0).sum() # same as tensor.sum(dim = 1).mean()

		return loss

if __name__ == "__main__":

	num_class = 10
	batch_size = 16
	loss = CrossEntropyLabelSmooth(num_class)
	labels = torch.randint(high = 10, size = [batch_size])
	scores = torch.randn(batch_size, num_class)

	res = loss(scores, labels)
	print(res)
