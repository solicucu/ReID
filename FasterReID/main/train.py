# -*- coding: utf-8 -*-
# @Author: solicucu

import sys
sys.path.append('..')
import os 
import argparse 
import logging 
import time 
import openpyxl as xl 
import torch 
import torch.nn as nn 

from config import cfg 
from torch.backends import cudnn
from utils import setup_logger, R1_mAP
from data import make_data_loader
from model import build_model, CDNetwork, CNetwork 
from optims import make_optimizer, make_lr_scheduler, make_loss 
from utils.utils import count_parameters, infer_count_parameters
from utils.metrics import * 
from utils.flop_benchmark import get_model_infos

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

def parse_config():

	# create the parser
	parser = argparse.ArgumentParser(description = "CDNet training")

	parser.add_argument("--config_file", default = '', help = "path to specify config file", type = str)

	#remainder parameters in a list
	parser.add_argument("opts", default = None, help = 'modify some value for the config file in command line', nargs = argparse.REMAINDER)

	args = parser.parse_args()

	if args.config_file != "":
		# use config file to update the default config value
		cfg.merge_from_file(args.config_file)
	# note that opts is a list 
	cfg.merge_from_list(args.opts)
	# cfg.freeze() if use this, the cfg can not be revised

	output_dir = cfg.OUTPUT.DIRS 
	if output_dir != "":
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
	else:
		print("ERROR:please specify an output path")
		exit(1)

	# config the logger 
	logger = setup_logger("CDNet", output_dir, 0, cfg.OUTPUT.LOG_NAME)

	use_gpu = cfg.MODEL.DEVICE == "cuda"
	if use_gpu:
		logger.info("Train with GPU: {}".format(cfg.MODEL.DEVICE_IDS))
	else:
		logger.info("Train with CPU")

	#print the all arguments
	logger.info(args)
	#read the config file
	if args.config_file != "":
		logger.info("load configuration file {}".format(args.config_file))

	# print the configuration
	logger.info("running with config:\n{}".format(cfg))

	if use_gpu:
		os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_IDS
	
	#this setup will facilitate the training
	cudnn.benchmark = True 

def train():

	# 1、make dataloader
	# prepare train,val img_info list, elem is tuple; 
	train_loader, val_loader, num_query, num_class = make_data_loader(cfg)
	
	# 2、make model
	model = build_model(cfg, num_class)

	# 3、 make optimizer
	optimizer = make_optimizer(cfg, model)

	# 4、 make lr_scheduler
	scheduler = make_lr_scheduler(cfg, optimizer)

	# 5、make loss 
	loss_fn = make_loss(cfg, num_class)

	# get parameters 
	device = cfg.MODEL.DEVICE
	use_gpu = device == "cuda"
	pretrained = cfg.MODEL.PRETRAIN_PATH != ""
	parallel = cfg.MODEL.PARALLEL

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

	logger = logging.getLogger("CDNet.train")
	size = count_parameters(model)
	values.append(format(size, '.2f'))
	values.append(model.final_planes)
	
	logger.info("the param number of the model is {:.2f} M".format(size))
	infer_size = infer_count_parameters(model)
	logger.info("the infer param number of the model is {:.2f}M".format(infer_size))

	shape = [1, 3]
	shape.extend(cfg.DATA.IMAGE_SIZE)
	
	# if cfg.MODEL.NAME == 'cdnet' :
	# 	infer_model = CDNetwork(num_class, cfg)
	# elif cfg.MODEL.NAME == 'cnet':
	# 	infer_model = CNetwork(num_class, cfg)
	# else:
	# 	infer_model = model 

	# for scaling experiment
	flops, _ = get_model_infos(model, shape)
	logger.info("the total flops number of the model is {:.2f} M".format(flops))
	
	logger.info("Starting Training CDNetwork")
	
	best_mAP, best_r1 = 0., 0.
	is_best = False
	avg_loss, avg_acc = RunningAverageMeter(),RunningAverageMeter()
	avg_time, global_avg_time = AverageMeter(), AverageMeter()

	if parallel:
		model = nn.DataParallel(model)
		
	if use_gpu:
		model = model.to(device)

	for epoch in range(epochs):
		
		scheduler.step()
		lr = scheduler.get_lr()[0]
		# if save epoch_num k, then run k+1 epoch next
		if pretrained and epoch < model.start_epoch:
			continue

		# rest the record
		model.train()
		avg_loss.reset()
		avg_acc.reset()
		avg_time.reset()

		for i, batch in enumerate(train_loader):

			t0 = time.time()
			imgs, labels = batch 

			if use_gpu:
				imgs = imgs.to(device)
				labels = labels.to(device)

			res = model(imgs)
		
			loss, acc = compute_loss_acc(res, labels, loss_fn)
			loss.backward()

			if grad_clip != 0:
				nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

			optimizer.step()
			optimizer.zero_grad()

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
		if (epoch + 1) % eval_period == 0 or (epoch + 1) in check_epochs:

			model.eval()
			metrics = R1_mAP(num_query, use_gpu = use_gpu)

			with torch.no_grad():
				for vi, batch in enumerate(val_loader):
					
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
			logger.info("checkpoint {} was saved".format(epoch + 1))

			if is_best:
				torch.save(model.state_dict(), ckpt_save_path + "best_ckpt.pth")
				logger.info("best_checkpoint was saved")
				is_best = False
		

	values.insert(1, format(global_avg_time.avg * 1000, '.2f'))
	values.append(format(infer_size, '.2f'))
	sheet.append(values)
	wb.save(result_path)
	logger.info("best_mAP:{:.2%}, best_r1:{:.2%}".format(best_mAP, best_r1))
	logger.info("Ending training CDNetwork")

def main():

	parse_config()
	train()

if __name__ == "__main__":

	main()