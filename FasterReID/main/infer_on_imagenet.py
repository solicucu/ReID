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
from utils import setup_logger
from data import imagenet_make_data_loader 
from model import build_model
from utils.utils import count_parameters, infer_count_parameters
from utils.metrics import * 
from utils.flop_benchmark import get_model_infos
"""
Args:
	res: a list of elem like [scores,feats]
	labels: ground truth of the imgs class 
	loss_fn: type of loss function to compute the loss 
Returns:
	loss: summation of each loss 
	acc: avg of each loss 
"""
def compute_loss_acc(res, labels, loss_fn = None):

	loss, acc = 0. , 0. 
	num = len(res)
	# for imagenet val 
	if loss_fn is None:
		for scores in res: 
			ac = (scores.max(1)[1] == labels).float().mean()
			acc += ac
		return loss, acc / num  


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

def test():
	logger = logging.getLogger('CDNet.test')
	# 1、make dataloader
	train_loader, val_loader, num_class = imagenet_make_data_loader(cfg, is_val = True)
	# print("num_query:{},num_class:{}".format(num_query,num_class))

	# 2、make model
	model = build_model(cfg, num_class)

	# load param
	ckpt_path = cfg.OUTPUT.DIRS + cfg.OUTPUT.CKPT_DIRS + cfg.TEST.BEST_CKPT 
	if os.path.isfile(ckpt_path):
		model.load_best_checkpoint(ckpt_path)
	else:
		logger.info("file: {} is not found".format(ckpt_path))
		exit(1)

	use_gpu = cfg.MODEL.DEVICE == 'cuda'
	device = cfg.MODEL.DEVICE_IDS

	size = count_parameters(model)
	logger.info("the param number of the model is {:.2f}M".format(size))

	flops, _ = get_model_infos(model, [1,3,224,244])
	logger.info("the total flops number of the model is {:.2f} M".format(flops))

	if cfg.MODEL.PARALLEL:
		model = nn.DataParallel(model)
	if use_gpu:
		model = model.cuda()

	model.eval()
	logger.info("begin eval the model")
	val_acc = AverageMeter()
	with torch.no_grad():

		for vi, batch in enumerate(val_loader):
			
			imgs, labels = batch

			if use_gpu:
				imgs = imgs.cuda()
				labels = labels.cuda()

			res = model(imgs)
			# acc = (scores.max(1)[1] == labels).float().mean()
			_, acc = compute_loss_acc(res, labels) 
			val_acc.update(acc)

		logger.info("final test acc is:{:.2%}".format(val_acc.avg))
		
	logger.info("Ending testing CDNetwork on imagenet")

def main():

	parse_config()
	test()

if __name__ == "__main__":

	main()