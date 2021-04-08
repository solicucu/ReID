# -*- coding: utf-8 -*-
# @Author: solicucu

import os 
import sys
import argparse
import logging
import torch
import time 
sys.path.append('..')
import torch.nn as nn 
from config import cfg 
from utils import setup_logger, R1_mAP
from data import make_data_loader
from extra import make_dataloader 
from extra import make_combine_dataloader
from model import build_model
from torch.backends import cudnn
from utils.utils import count_parameters, infer_count_parameters 
from utils.flop_benchmark import get_model_infos


def parse_config():

	parser = argparse.ArgumentParser(description = 'CDNet inference')

	parser.add_argument("--config_file", default = "", help = "path to specified config file", type = str)

	parser.add_argument("opts", default = None, help = "modify some value in config file", nargs = argparse.REMAINDER)

	args = parser.parse_args()

	if args.config_file != "":
		cfg.merge_from_file(args.config_file)
	cfg.merge_from_list(args.opts)

	cfg.freeze()

	output_dir = cfg.OUTPUT.DIRS 
	if output_dir != "":
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
	else:
		print("ERROR: please specify an output path")
		exit(1)

	logger = setup_logger("CDNet", output_dir,0,cfg.OUTPUT.LOG_NAME)

	use_gpu = cfg.MODEL.DEVICE == 'cuda'

	if use_gpu:
		logger.info("Test with GPU: {}".format(cfg.MODEL.DEVICE_IDS))
	else:
		logger.info("Test with CPU")

	logger.info(args)
	if args.config_file != "":
		logger.info("load configuratioin file {}".format(args.config_file))

	logger.info("test with config:\n{}".format(cfg))

	if use_gpu:
		os.environ["CUDA_VISIBLE_DEVICE"] =cfg.MODEL.DEVICE_IDS
	cudnn.benchmark = True


def test():

	logger = logging.getLogger('CDNet.test')

	# prepare dataloader
	train_loader, val_loader, num_query, num_class = make_data_loader(cfg)
	# prepare model
	model = build_model(cfg, num_class)

	infer_size = infer_count_parameters(model)
	logger.info("the infer param number of the model is {:.2f}M".format(infer_size))

	shape = [1, 3]
	shape.extend(cfg.DATA.IMAGE_SIZE)
	flops, _ = get_model_infos(model, shape)
	logger.info("the total flops is: {:.2f} M".format(flops))
	
	# load param
	ckpt_path = cfg.OUTPUT.DIRS + cfg.OUTPUT.CKPT_DIRS + cfg.TEST.BEST_CKPT 
	
	if os.path.isfile(ckpt_path):
		model.load_best_checkpoint(ckpt_path)
	else:
		logger.info("file: {} is not found".format(ckpt_path))
		exit(1)

	use_gpu = cfg.MODEL.DEVICE == 'cuda'
	if cfg.MODEL.PARALLEL:
		model = nn.DataParallel(model)
	if use_gpu:
		model = model.cuda()
	model.eval()
	metrics = R1_mAP(num_query, use_gpu = use_gpu)

	with torch.no_grad():
		begin = time.time()
		for batch in val_loader:
			imgs, pids, camids = batch 

			if use_gpu:
				imgs = imgs.cuda()
			feats = model(imgs)
			metrics.update((feats, pids, camids))
		end1 = time.time()
		cmc, mAP = metrics.compute()
		end2 = time.time()
		logger.info("extract feature time is:{:.2f} s".format(end1 - begin))
		logger.info("match time is:{:.2f} s".format(end2 - end1))

		logger.info("test result as follows")
		logger.info("mAP:{:.2%}".format(mAP))
		for r in [1,5,10]:
			logger.info("CMC cure, Rank-{:<3}:{:.2%}".format(r, cmc[r-1]))

		print("test is endding")


def extract_features(save_path = './', num_pids = 20):

	logger = logging.getLogger('CDNet.test')

	# prepare dataloader
	# param num_pids=5, batch_size = 64, num_workers = 4
	dataloader, num_class = make_dataloader(num_pids = num_pids, batch_size = cfg.SOLVER.BATCH_SIZE, num_workers = cfg.DATALOADER.NUM_WORKERS)
	# prepare model
	model = build_model(cfg, num_class)

	infer_size = infer_count_parameters(model)
	logger.info("the infer param number of the model is {:.2f}M".format(infer_size))

	# load param
	ckpt_path = cfg.OUTPUT.DIRS + cfg.OUTPUT.CKPT_DIRS + cfg.TEST.BEST_CKPT 
	if os.path.isfile(ckpt_path):
		model.load_best_checkpoint(ckpt_path)
	else:
		logger.info("file: {} is not found".format(ckpt_path))
		exit(1)

	use_gpu = cfg.MODEL.DEVICE == 'cuda'
	if cfg.MODEL.PARALLEL:
		model = nn.DataParallel(model)
	if use_gpu:
		model = model.cuda()

	model.eval()
	tri_features = []
	fc_features = []
	label_list = []
	pid_list = []

	with torch.no_grad():
		for batch in dataloader:
			imgs, labels, pids = batch 

			if use_gpu:
				imgs = imgs.cuda()
			feats = model(imgs)
			# features.append(feats)
			tri_features.append(feats[0])
			fc_features.append(feats[1])
			label_list.extend(labels)
			pid_list.extend(pids)

	tri_features = torch.cat(tri_features, dim = 0).cpu()
	fc_features = torch.cat(fc_features, dim = 0).cpu()
	tri_file_name = save_path + 'bnneck_tri_feats_20.feats'
	fc_file_name = save_path + 'bnneck_fc_feats_20.feats'

	with open(save_path + 'bnneck_label_pid.txt', 'w') as f: 
		for i in range(len(label_list)):
			f.write("{} {}\n".format(label_list[i],pid_list[i]))

	torch.save(tri_features, tri_file_name)
	torch.save(fc_features, fc_file_name)

	logger.info("extract features is endding, total images:{} and identities:{}".format(tri_features.size(0), num_pids))

def extract_fmap(save_path = './'):

	logger = logging.getLogger('CDNet.test')

	# prepare dataloader
	# param num_pids=5, batch_size = 64, num_workers = 4
	dataloader, num_class = make_combine_dataloader()
	# prepare model
	model = build_model(cfg, num_class)

	infer_size = infer_count_parameters(model)
	logger.info("the infer param number of the model is {:.2f}M".format(infer_size))

	# load param
	ckpt_path = cfg.OUTPUT.DIRS + cfg.OUTPUT.CKPT_DIRS + cfg.TEST.BEST_CKPT 
	if os.path.isfile(ckpt_path):
		model.load_best_checkpoint(ckpt_path)
	else:
		logger.info("file: {} is not found".format(ckpt_path))
		exit(1)

	use_gpu = cfg.MODEL.DEVICE == 'cuda'
	if cfg.MODEL.PARALLEL:
		model = nn.DataParallel(model)
	if use_gpu:
		model = model.cuda()

	model.eval()
	fmap1 = []
	fmap2 = []
	fmap3 = []
	label_list = []

	with torch.no_grad():
		for batch in dataloader:
			imgs, labels, pids = batch 

			if use_gpu:
				imgs = imgs.cuda()
			feats = model(imgs)
			# features.append(feats)
			fmap1.append(feats[0])
			fmap2.append(feats[1])
			fmap3.append(feats[2])
			label_list.extend(labels)

	fmap1 = torch.cat(fmap1, dim = 0).cpu()
	fmap2 = torch.cat(fmap2, dim = 0).cpu()
	fmap3 = torch.cat(fmap3, dim = 0).cpu()

	fmap1_file_name = save_path + 'fmap1.fmap'
	fmap2_file_name = save_path + 'fmap2.fmap'
	fmap3_file_name = save_path + 'fmap3.fmap'

	with open(save_path + 'labels.txt', 'w') as f: 
		for i in range(len(label_list)):
			f.write("{}\n".format(label_list[i]))

	torch.save(fmap1, fmap1_file_name)
	torch.save(fmap2, fmap2_file_name)
	torch.save(fmap3, fmap3_file_name)

	logger.info("extract features is endding, total images:{}".format(fmap1.size(0)))

if __name__ == "__main__":

	parse_config()
	test()
	# extract_features()
	# extract_fmap() # for visualization