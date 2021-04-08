# -*- coding: utf-8 -*-
# @Author: solicucu

# root = "D:/project/Paper/papercode/FasterReID/" # for pc
root = "/home/hanjun/solicucu/ReID/FasterReID/" # for server
import sys 
sys.path.append(root)
import os 
import torch 
import numpy as np 
import logging 
import time 
import torch.nn as nn 
import openpyxl as xl 

from configs import _C as cfg 
from utils import setup_logger, R1_mAP
from torch.backends import cudnn
from data import darts_make_data_loader
from cnet_search import CNetwork 
from optims import make_optimizer, make_lr_scheduler, darts_make_loss
from utils.utils import count_parameters
from utils.metrics import * 

"""
Args:
	res: a list of elem like [scores, feats]
	labels: ground truth of the imgs class 
	loss_fn: type of loss function to compute the loss 
Returns:
	loss: summation of each loss 
	acc: avg of each loss 
"""
def compute_loss_acc(res, labels, loss_fn):

	loss, acc = 0. , 0. 
	num = len(res)
	
	for scores, feats in res:
		ls = loss_fn(scores, feats, labels)
		loss += ls 

		ac = (scores.max(1)[1] == labels).float().mean()
		acc += ac

	return loss, acc / num

def set_config():
	
	output_dir = cfg.OUTPUT.DIRS 
	if output_dir != "":
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
	else:
		print("ERROR:please specify an output path")
		exit(1)

	# config the logger 
	logger = setup_logger("CNet_Search", output_dir, 0, cfg.OUTPUT.LOG_NAME)

	use_gpu = cfg.MODEL.DEVICE == "cuda"
	if use_gpu:
		logger.info("Train with GPU: {}".format(cfg.MODEL.DEVICE_IDS))
	else:
		logger.info("Train with CPU")

	if use_gpu:
		os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_IDS
	cudnn.benchmark = True

	# print the configuration
	logger.info("running with config:\n{}".format(cfg))

	# init rand seed 
	seed = cfg.SOLVER.SEED
	np.random.seed(seed)
	torch.manual_seed(seed)
	if use_gpu:
		torch.cuda.manual_seed(seed)

def train():
	
	use_gpu = cfg.MODEL.DEVICE == "cuda"
	# 1、make dataloader 
	train_loader, val_loader, test_loader, num_query, num_class =  darts_make_data_loader(cfg)
	# print(num_query, num_class)
	
	# 2、make model
	model = CNetwork(num_class, cfg)
	# tensor = torch.randn(2, 3, 256, 128)
	# res = model(tensor)
	# print(res[0].size()) [2, 751]

	# 3、make optimizer
	optimizer = make_optimizer(cfg, model)
	# make architecture optimizer
	arch_optimizer = torch.optim.Adam(model._arch_parameters(),
			lr = cfg.SOLVER.ARCH_LR, betas = (0.5, 0.999), weight_decay = cfg.SOLVER.ARCH_WEIGHT_DECAY)

	# 4、make lr scheduler
	lr_scheduler = make_lr_scheduler(cfg, optimizer)
	# make lr scheduler
	arch_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(arch_optimizer, [80, 160], 0.1)

	# 5、make loss 
	loss_fn = darts_make_loss(cfg)

	# get parameters
	device = cfg.MODEL.DEVICE
	use_gpu = device == "cuda"
	pretrained = cfg.MODEL.PRETRAINED != ""

	log_period = cfg.OUTPUT.LOG_PERIOD
	ckpt_period = cfg.OUTPUT.CKPT_PERIOD
	eval_period = cfg.OUTPUT.EVAL_PERIOD
	output_dir = cfg.OUTPUT.DIRS
	ckpt_save_path = output_dir + cfg.OUTPUT.CKPT_DIRS
	
	epochs = cfg.SOLVER.MAX_EPOCHS
	batch_size = cfg.SOLVER.BATCH_SIZE
	grad_clip = cfg.SOLVER.GRAD_CLIP

	batch_num = len(train_loader)
	log_iters = batch_num // log_period 

	if not os.path.exists(ckpt_save_path):
		os.makedirs(ckpt_save_path)

	# create *_result.xlsx
	# save the result for analyze
	name = (cfg.OUTPUT.LOG_NAME).split(".")[0] + ".xlsx"
	result_path = cfg.OUTPUT.DIRS + name

	wb = xl.Workbook()
	sheet = wb.worksheets[0]
	titles = ['size/M','speed/ms','final_planes', 'acc', 'mAP', 'r1', 'r5', 'r10', 'loss',
			  'acc', 'mAP', 'r1', 'r5', 'r10', 'loss','acc', 'mAP', 'r1', 'r5', 'r10', 'loss']
	sheet.append(titles)
	check_epochs = [40, 80, 120, 160, 200, 240, 280, 320, 360, epochs]
	values = []

	logger = logging.getLogger("CNet_Search.train")
	size = count_parameters(model)
	values.append(format(size, '.2f'))
	values.append(model.final_planes)
	
	logger.info("the param number of the model is {:.2f} M".format(size))

	logger.info("Starting Search CNetwork")

	best_mAP, best_r1 = 0., 0.
	is_best = False
	avg_loss, avg_acc = RunningAverageMeter(),RunningAverageMeter()
	avg_time, global_avg_time = AverageMeter(), AverageMeter()

	if use_gpu:
		model = model.to(device)

	if pretrained:
		logger.info("load self pretrained chekpoint to init")
		model.load_pretrained_model(cfg.MODEL.PRETRAINED)
	else:
		logger.info("use kaiming init to init the model")
		model.kaiming_init_()

	for epoch in range(epochs):

		lr_scheduler.step()
		lr = lr_scheduler.get_lr()[0]
		# architect lr.step
		arch_lr_scheduler.step()

		# if save epoch_num k, then run k+1 epoch next
		if pretrained and epoch < model.start_epoch:
			continue

		# print(epoch)
		# exit(1)
		model.train()
		avg_loss.reset()
		avg_acc.reset()
		avg_time.reset()

		for i, batch in enumerate(train_loader):
			
			t0 = time.time()
			imgs, labels = batch
			val_imgs, val_labels = next(iter(val_loader))

			if use_gpu:
				imgs = imgs.to(device)
				labels = labels.to(device)
				val_imgs = val_imgs.to(device)
				val_labels = val_labels.to(device)

			# 1、 update the weights
			optimizer.zero_grad()
			res = model(imgs)

			# loss = loss_fn(scores, feats, labels)
			loss, acc = compute_loss_acc(res, labels, loss_fn)
			loss.backward()

			if grad_clip != 0:
				nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

			optimizer.step() 

			# 2、update the alpha
			arch_optimizer.zero_grad()
			res = model(val_imgs)

			val_loss, val_acc = compute_loss_acc(res, val_labels, loss_fn)
			val_loss.backward()
			arch_optimizer.step()

			# compute the acc 
			# acc = (scores.max(1)[1] == labels).float().mean()

			t1 = time.time()
			avg_time.update((t1 - t0) / batch_size)
			avg_loss.update(loss)
			avg_acc.update(acc)

			# log info
			if (i+1) % log_iters == 0:
				logger.info("epoch {}: {}/{} with loss is {:.5f} and acc is {:.3f}".format(
					epoch+1, i+1, batch_num, avg_loss.avg, avg_acc.avg))

		logger.info("end epochs {}/{} with lr: {:.5f} and avg_time is: {:.3f} ms".format(epoch+1, epochs, lr, avg_time.avg * 1000))
		global_avg_time.update(avg_time.avg)

		# test the model
		if (epoch + 1) % eval_period == 0:

			model.eval()
			metrics = R1_mAP(num_query, use_gpu = use_gpu)

			with torch.no_grad():
				for vi, batch in enumerate(test_loader):
					# break
					# print(len(batch))
					imgs, labels, camids = batch
					if use_gpu:
						imgs = imgs.to(device)

					feats = model(imgs)
					metrics.update((feats, labels, camids))

				#compute cmc and mAP
				cmc, mAP = metrics.compute()
				logger.info("validation results at epoch {}".format(epoch + 1))
				logger.info("mAP:{:2%}".format(mAP))
				for r in [1,5,10]:
					logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r-1]))

				# determine whether current model is the best
				if mAP > best_mAP:
					is_best = True
					best_mAP = mAP
					logger.info("Get a new best mAP")
				if cmc[0] > best_r1:
					is_best = True
					best_r1 = cmc[0]
					logger.info("Get a new best r1")

				# add the result to sheet
				if (epoch + 1) in check_epochs:
					val = [avg_acc.avg, mAP, cmc[0], cmc[4], cmc[9]]
					change = [format(v * 100, '.2f') for v in val]
					change.append(format(avg_loss.avg, '.3f'))
					values.extend(change)
		
		# whether to save the model
		if (epoch + 1) % ckpt_period == 0 or is_best:
			torch.save(model.state_dict(), ckpt_save_path + "checkpoint_{}.pth".format(epoch + 1))
			model._parse_genotype(file = ckpt_save_path + "genotype_{}.json".format(epoch + 1))
			logger.info("checkpoint {} was saved".format(epoch + 1))

			if is_best:
				torch.save(model.state_dict(), ckpt_save_path + "best_ckpt.pth")
				model._parse_genotype(file = ckpt_save_path + "best_genotype.json")
				logger.info("best_checkpoint was saved")
				is_best = False
		# exit(1)

	values.insert(1, format(global_avg_time.avg * 1000, '.2f'))
	sheet.append(values)
	wb.save(result_path)

	logger.info("Ending Search CNetwork")

def main():
	set_config()
	train()

if __name__ == "__main__":

	main()
