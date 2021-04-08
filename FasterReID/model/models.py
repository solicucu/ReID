# -*- coding: utf-8 -*-
# @Author: solicucu

import logging
import glob  
import torch 
import torch.nn as nn 
import torch.nn.init as init
from .backbone import OSNetwork 
from .backbone import CNetwork
from .backbone import CDNetwork
from .backbone import MobileNetV2
from .backbone import STDNetwork 
from .backbone import GDASNet 
from .head import STDNeck 
from .head import BLNeck 
from .head import FBLNeck 

logger = logging.getLogger("CDNet.model")

class BaseNet(nn.Module):

	def __init__(self, num_class, cfg = None):

		super(BaseNet, self).__init__()

		model_name = cfg.MODEL.NAME 
		neck_name = cfg.MODEL.NECK_TYPE
		bnneck = cfg.MODEL.USE_BNNECK

		self.self_ckpt = cfg.MODEL.PRETRAIN_PATH
		self.imagenet_ckpt = cfg.MODEL.IMAGENET_CKPT 
		self.start_epoch = 0                                            
		self.classification = cfg.DATA.DATASET in ['cifar10', 'cifar100', 'cifar100_combine', 'imagenet']

		if model_name == "osnet":

			self.base = OSNetwork(num_class, pretrained = False, loss = cfg.SOLVER.LOSS_NAME)
			self.final_planes = self.base.final_planes 

		elif model_name == "cnet":

			self.base = CNetwork(num_class, cfg)
			self.final_planes = self.base.final_planes 

		elif model_name == "mobilenetv2":

			self.base = MobileNetV2(width_mult = cfg.MODEL.WIDTH_MULT, before_gap = True)
			self.final_planes = self.base.final_planes 
		
		elif model_name == "cdnet":

			self.base = CDNetwork(num_class, cfg)
			self.final_planes = self.base.final_planes 

		elif model_name == 'stdnet':

			self.base = STDNetwork(num_class, cfg)
			self.final_planes = self.base.final_planes 

		elif model_name == 'gdasnet':

			self.base = GDASNet(num_class, cfg)
			self.final_planes = self.base.final_planes 
				
		else:

			raise RuntimeError("{} not implement".format(model_name))

		logger.info("final planes: {}".format(self.base.planes))

		if neck_name == "stdneck":

			self.neck = STDNeck(num_class, self.final_planes, dropout = cfg.TRICKS.DROPOUT, fc_dims = cfg.MODEL.FC_DIMS, use_bnneck = bnneck)
			self.final_planes = self.neck.final_planes 

		elif neck_name == "blneck":

			self.neck = BLNeck(num_class, self.final_planes, dropout = cfg.TRICKS.DROPOUT, fc_dims = cfg.MODEL.FC_DIMS, classification = self.classification )
			self.final_planes = self.neck.final_planes

		elif neck_name == "fblneck":

			self.neck = FBLNeck(num_class, self.base.planes, dropout = cfg.TRICKS.DROPOUT, fc_dims = cfg.MODEL.FC_DIMS, classification = self.classification)
			self.final_planes = self.neck.final_planes
			
		elif neck_name == "none":
			# some network no need the neck 
			self.neck = None
		
		else:
			raise RuntimeError("{} not implement".format(neck_name))

		if self.self_ckpt != "":
			
			logger.info("load the latest checkpoint from self training")
			self.load_latest_state_dict()

		else:
			self.kaiming_init_()
			# pass 
		if self.imagenet_ckpt != '':
			self.load_imagenet_state_dict()
			

	def forward(self, x):

		res = self.base(x)

		if self.neck is not None:
			res = self.neck(res)

		return res

	def kaiming_init_(self):

		logger.info("use kaiming init the model")
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				init.kaiming_normal_(m.weight)
				if m.bias is not None:
					init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				if m.weight is not None:
					init.constant_(m.weight, 1)
				if m.bias is not None:
					init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				init.normal_(m.weight, std = 1e-3)
				if m.bias is not None:
					init.constant_(m.bias, 0)
					
	def load_imagenet_state_dict(self):

		logger.info("load the self-trained imagenet ckpt to init the model")

		state_dict = torch.load(self.imagenet_ckpt)
		self_state_dict = self.state_dict()
		# remove the key with classifier 

		for key in self_state_dict:
			if 'classifier' in key:
				continue 
			self_state_dict[key].data.copy_(state_dict[key].data)

		logger.info("end of loading checkpoint from {}".format(self.imagenet_ckpt)) 

	def load_best_checkpoint(self, path):

		state_dict = torch.load(path) 
		# self.load_state_dict(state_dict)
		# or 
		self_state_dict = self.state_dict()
		for key in self_state_dict:
			self_state_dict[key].data.copy_(state_dict[key].data)

		logger.info("load the best checkpoint from {}".format(path))

	def load_latest_state_dict(self):

		logger.info("load the latest checkpoint")
		# checkpoint
		ckpt_list = glob.glob(self.self_ckpt + "checkpoint_*")
		ckpt_list = sorted(ckpt_list)
		# print(ckpt_list)
		# exit(1)
		ckpt_name = ckpt_list[-1]
		# print(ckpt_name)
		num = int(ckpt_name.split("/")[-1].split("_")[1].split(".")[0])
		self.start_epoch = num

		#self.load_state_dict(torch.load(ckpt_name)) 
		# or
		self_state_dict = self.state_dict()
		state_dict = torch.load(ckpt_name)
		for key in self_state_dict:
			self_state_dict[key].data.copy_(state_dict[key].data)
		logger.info("load checkpoint from {}".format(ckpt_name))