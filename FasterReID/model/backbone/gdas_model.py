import torch
import torch.nn as nn
import torch.nn.functional as F
from factory.gdas import * 
from torch.autograd import Variable
from .genotypes import PRIMITIVES
from .genotypes import *
import numpy as np
import json
import random
import logging 
logger = logging.getLogger("CDNet")

class Network(nn.Module):
  def __init__(self, num_class, cfg):
    super(Network, self).__init__()
    self.num_classes = num_class
    
    # self.planes = cfg.MODEL.PLANES    # 64
    # data type is different from cdnet
    # self.nodes = cfg.MODEL.NODES      # 4
    # self.layers = cfg.MODEL.LAYERS    # 6
    # self.stages = cfg.MODEL.STAGES    # 3
    # set the param here
    self.in_planes = 64 
    self.nodes = 4 
    self.layers = 6
    self.stages = 3 
    self.extract_stages_feats = cfg.MODEL.NECK_TYPE == 'fblneck'
    self.planes = []
    logger.info("model info: in_planes:{}, nodes:{}, layers:{}, stages:{}".format(self.in_planes, self.nodes, self.layers, self.stages))
    
    self.fc_dims = cfg.MODEL.FC_DIMS   # 512
    self.fc_num = len(self.fc_dims)
    self.tau = 10
    self.before_gap = True 
    self.dropout = 0.2
    self.genotype = genotype_model 

    C_curr = self.in_planes
    self.stem0 = nn.Sequential(
      nn.Conv2d(3, C_curr // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C_curr // 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(C_curr // 2, C_curr, 3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(C_curr),
    )

    self.stem1 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Conv2d(C_curr, C_curr, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C_curr),
    )

    assert len(self.genotype) == self.layers

    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C_curr
    # reduction_layer = [0, self.layers//self.stages, self.layers//self.stages*2]
    reduction_layer = [self.layers//self.stages, self.layers//self.stages*2]
    self.cells = nn.ModuleList()
    reduction = True
    reduction_prev = True
    for j, genotype in enumerate(self.genotype):
      if j in reduction_layer:
        C_curr *= 2
        self.planes.append(C_curr)
        reduction_prev = reduction
        reduction = True
      else:
        reduction_prev = reduction
        reduction = False
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, C_curr * cell.multiplier

    self.planes = [i*4 for i in self.planes]
    self.planes[-2] = self.planes[-1]
    
    self.conv1x1 = Conv1x1BNReLU(C_prev, C_prev)
    self.gap = nn.AdaptiveAvgPool2d(1)
    self.final_planes = C_prev

    if self.dropout > 0:
      self.drop = nn.Dropout(self.dropout)

    if self.fc_num > 0 and not self.before_gap:
      self.fc = self._make_fc_layers(C_prev)
      self.final_planes = self.fc_dims[-1]

    # classifier
    if not self.before_gap:
      self.classifier = nn.Linear(self.final_planes, num_class)

  def _make_fc_layers(self, in_planes):

    layers = []
    in_dim = in_planes

    for dim in self.fc_dims:

      layers.append(nn.Linear(in_dim, dim))
      layers.append(nn.BatchNorm1d(dim))
      layers.append(nn.ReLU(inplace = True))

      in_dim = dim

    return nn.Sequential(*layers)

  def kaiming_init_(self):

    # print("use kaiming init")
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

  def train_fix_bn(self, mode=True):
    # freeze BN mean and std
    for module in self.children():
      # for module in self.modules():
      if isinstance(module, nn.BatchNorm2d):
        module.train(False)
      else:
        module.train()

  def forward(self, input):
    s0 = self.stem0(input)
    s1 = self.stem1(s0)
    stages = [1, 4, 5]
    feature_maps = []
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1)
      # print('layer {}, shape: {}'.format(i, s1.shape))
      if self.extract_stages_feats and i in stages:
        feature_maps.append(s1)

    if self.extract_stages_feats:
      return feature_maps 

    x = self.conv1x1(s1)
    if self.before_gap:
      return x

    x = self.gap(x)
    batch = x.size(0)
    feat = x.view(batch, -1)
    if self.dropout > 0:
      feat = self.drop(feat)
    if self.fc_num > 0:
      feat_fc = self.fc(feat)
    else:
      feat_fc = feat
    if not self.training:
      return feat
    score = self.classifier(feat_fc)
    return [[score, feat]]


class Cell(nn.Module):
  def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction
    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    op_names, indices = zip(*genotype.cell)
    concat = genotype.cell_concat
    self._compile(C, op_names, indices, concat, reduction)

  def _compile(self, C_curr, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._nodes = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()

        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C_curr, stride, True)
            self._ops += [op]
        self._indices = indices

  def forward(self, s0, s1, drop_path_prob=0.3):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    for i in range(self._nodes):
      h1 = states[self._indices[2 * i]]
      h2 = states[self._indices[2 * i + 1]]
      op1 = self._ops[2 * i]
      op2 = self._ops[2 * i + 1]
      h1 = op1(h1)
      h2 = op2(h2)
      s = h1 + h2
      states += [s]

    return torch.cat([states[i] for i in self._concat], dim=1)
