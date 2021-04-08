# -*- coding: utf-8 -*-
# @Author: solicucu

from yacs.config import CfgNode as CN 

_C = CN()

#------------------------------
# MODEL
# config for model paramter
#------------------------------

_C.MODEL = CN()
# using gpu or cpu for the traing
_C.MODEL.DEVICE = 'cpu'
# specify the gpu to be used if use gpu
_C.MODEL.DEVICE_IDS = '0'
# name of the backbone
_C.MODEL.NAME = 'cdnet'
# load the specified checkpoint for the model, self pretrained
_C.MODEL.PRETRAIN_PATH = ''
# load the imagenet checkpoint which we trained
_C.MODEL.IMAGENET_CKPT = ''
# whether separate the feature use for triplet loss and softmax loss 
_C.MODEL.USE_BNNECK = False
# whether use DataParallel
_C.MODEL.PARALLEL = False
# for scaling the model width x1 x2 x1.5
_C.MODEL.WIDTH_MULT = 1.
# specify the num of cell in each stage
_C.MODEL.STAGES = [2, 2, 2, 2]
# specify the channel in ecch stage
_C.MODEL.PLANES = [32, 64, 128, 256]
# model genotype path for CDNet/CNet 
_C.MODEL.GENOTYPE = 'best_genotype.json'
# neck types
_C.MODEL.NECK_TYPE = "stdneck"
# whether use adaption fusion
_C.MODEL.ADAPTION_FUSION = False
# dim list for construct fc before classify
_C.MODEL.FC_DIMS = []

#--------------------------------
# DATA
# preprocess the data
#--------------------------------

_C.DATA = CN()
# which dataset to be use for training
_C.DATA.DATASET = "market1501"
# path to dataset
_C.DATA.DATASET_DIR = "D:/project/data/"
# _C.DATA.DATASET_DIR = "/home/share/solicucu/data/"
# size of the image during the training
_C.DATA.IMAGE_SIZE = [256,128]
# the probobility for random image horizontal flip
_C.DATA.HF_PROB = 0.5
# the probobility for random image erasing
_C.DATA.RE_PROB = 0.5
# rgb means used for image normalization
_C.DATA.MEAN = [0.485, 0.456, 0.406]
# rgb stds used for image normalization
_C.DATA.STD = [0.229, 0.224, 0.225]
# value of padding size
_C.DATA.PADDING = 10
# cutout # infer cutout=16
_C.DATA.CUTOUT = 0

#--------------------------------
# DATALOADER
#--------------------------------

_C.DATALOADER = CN()
# number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
# types of Sampler for data loading
_C.DATALOADER.SAMPLER = 'triplet'
# number of instance for single person
_C.DATALOADER.NUM_INSTANCE = 4


#--------------------------------
# SOLVER
#--------------------------------

_C.SOLVER = CN()
# total number of epoch for training
_C.SOLVER.MAX_EPOCHS = 120
# number of images per batch
_C.SOLVER.BATCH_SIZE = 64

# learning rate
# the initial learning
_C.SOLVER.BASE_LR = 0.025
# the period for lerning decay for StepLR
_C.SOLVER.LR_DECAY_PERIOD = 10
# learning rate decay fator
_C.SOLVER.LR_DECAY_FACTOR = 0.1
# lr scheduler [StepLR, ConsineAnnealingLR, WarmupMultiStepLR]
_C.SOLVER.LR_SCHEDULER_NAME = "StepLR" 
# min_lr for ConsineAnnealingLR
_C.SOLVER.LR_MIN = 0.001 

# for warmupMultiStepLR 
# at which epoch change the lr
_C.SOLVER.MILESTONES = [40, 70]
# lr list for multistep
_C.SOLVER.LR_LIST = [3.5e-4, 3.5e-5, 3.5e-6]
# coefficient for linear warmup 
_C.SOLVER.GAMA = 0.
# use to calculate the start lr, init_lr = base_lr * warmup_factor
_C.SOLVER.WARMUP_FACTOR = 1.
# how many epoch to warmup, 0 denote do not use warmup 
_C.SOLVER.WARMUP_ITERS = 0
# method for warmup 
_C.SOLVER.WARMUP_METHOD = 'linear'

# optimizer
# the name of the optimizer
_C.SOLVER.OPTIMIZER_NAME = "SGD"
# momentum for SGD
_C.SOLVER.MOMENTUM = 0.9
# weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005

# loss
# loss type:softmax, triplet , softmax_triplet
_C.SOLVER.LOSS_NAME = "softmax"
# the margin for triplet loss
_C.SOLVER.TRI_MARGIN = 0.3
# clip the gradient if grad_clip is not zero
_C.SOLVER.GRAD_CLIP = 0.


#--------------------------------
# OUTPUT
#--------------------------------

_C.OUTPUT = CN()

# path to output
_C.OUTPUT.DIRS = "D:/project/data/ReID/ReIDModels/cdnet/market1501/"
# path to save the checkpoint
_C.OUTPUT.CKPT_DIRS = "checkpoints/cdnet_fblneck/"
# specify a name for log
_C.OUTPUT.LOG_NAME = "log_cdnet_fblneck.txt"
# the period for log
_C.OUTPUT.LOG_PERIOD = 10
# the period for saving the checkpoint
_C.OUTPUT.CKPT_PERIOD = 10
# the period for validatio
_C.OUTPUT.EVAL_PERIOD = 10


#--------------------------------
# TRICKS
# set some tricks here
#--------------------------------

# tircks 
_C.TRICKS = CN()
# use the label smooth to prevent overfiting
_C.TRICKS.LABEL_SMOOTH = False
# specify the dropout probability
_C.TRICKS.DROPOUT = 0.


#--------------------------------
# TEST
#--------------------------------
_C.TEST = CN()
# batch size for test
_C.TEST.IMGS_PER_BATCH = 128
# whether feature is normalized before test
_C.TEST.FEAT_NORM = 'yes'
# the name of best checkpoint for test
_C.TEST.BEST_CKPT = ''
