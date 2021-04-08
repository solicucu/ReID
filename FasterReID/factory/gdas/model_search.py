import torch
import torch.nn as nn
import torch.nn.init as init 
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from configs import _C as cfg
from genotypes import Genotype_stage
import numpy as np
import json
import random


class Network(nn.Module):
  def __init__(self, num_class, cfg):
    super(Network, self).__init__()
    self.planes = cfg.MODEL.PLANES    # 64
    self.num_classes = num_class
    self.nodes = cfg.MODEL.NODES      # 4
    self.layers = cfg.MODEL.LAYERS    # 6
    self.stages = cfg.MODEL.STAGES    # 3
    self.multiplier = cfg.MODEL.MULTIPLIER  # 4
    self.fc_dims = cfg.MODEL.FC_DIMS   # 512
    self.use_bnneck = cfg.MODEL.USE_BNNECK
    self.fc_num = len(self.fc_dims)
    self.tau = 10
    self.before_gap = False
    self.dropout = 0.2

    C_curr = self.planes
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

    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C_curr
    reduction_layer = [0, self.layers//self.stages, self.layers//self.stages*2]
    self.cells = nn.ModuleList()
    self.arch = []
    reduction = True
    reduction_prev = True
    for j in range(self.layers):
      if j in reduction_layer:
        C_curr *= 2
        reduction_prev = reduction
        reduction = True
      else:
        reduction_prev = reduction
        reduction = False
      cell = Cell(self.multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, self.nodes)
      self.arch.append(cell.arch_parameters())
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, C_curr * self.multiplier

    self.conv1x1 = Conv1x1BNReLU(C_prev, C_prev)
    self.gap = nn.AdaptiveAvgPool2d(1)
    self.final_planes = C_prev
    # print(self.final_planes)
    # exit(1)
    if self.dropout > 0:
      self.drop = nn.Dropout(self.dropout)

    if self.fc_num > 0:
      self.fc = self._make_fc_layers(C_prev)
      self.final_planes = self.fc_dims[-1]

    if self.use_bnneck:
      self.bnneck = nn.BatchNorm1d(self.final_planes)
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
    for i, cell in enumerate(self.cells):
      alphs_curr = self.arch[i]
      while True:
        gumbels = -torch.empty_like(alphs_curr).exponential_().log()
        logits = (alphs_curr.log_softmax(dim=1) + gumbels) / self.tau
        probs = F.softmax(logits, dim=1)
        # probs = F.softmax(alphs_curr, dim = 1) # for no gumbel
        index = probs.max(-1, keepdim=True)[1]
        one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        # one_h = torch.zeros_like(probs).scatter_(-1, index, 1.0) # for no gumbel
        weights = one_h - probs.detach() + probs
        if (torch.isinf(gumbels).any()) or (torch.isinf(probs).any()) or (torch.isnan(probs).any()):
          continue
        else: break
       

      s0, s1 = s1, cell.forward_hard(s0, s1, weights, index)
      # print('layer {}, shape: {}'.format(i, s1.shape))

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

  def _arch_parameters(self):

    return self.arch 

  def genotype(self):
    genotypes = []
    for cell in self.cells:
      genotypes.append(cell.genotype())
    return genotypes

  def set_tau(self, tau):
    self.tau = tau

  def load_arch(self, arc_np):
    arc = []
    for i in range(self._stages):
      self.stages[i].load_arch(arc_np[i])
      a = torch.from_numpy(arc_np[i])
      arc.append(a)
    self.arch = arc

  def fix_arch(self):
    for arc in self.arch:
      arc.requires_grad = False

  def _parse_genotype(self, file = "./genotype.json"):
    genotypes = self.genotype()
    name = file.split(".")[0] + ".txt"
    with open(name, 'w') as f:
      for i, genotype in enumerate(genotypes):
        print('{}th, {}'.format(i, genotype))
        f.write('{}th, {}'.format(i, genotype))
    geno = {}
    # geno['genotype'] = {}
    geno['alphas'] = {}
    for i, genotype in enumerate(genotypes):
      # geno['genotype'][i] = list(genotype._asdict())
      geno['alphas'][i] = self.arch[i].cpu().detach().numpy().tolist()

    json_data = json.dumps(geno, indent = 4)
    with open(file, 'w') as f:
      f.write(json_data)


class Cell(nn.Module):
  def __init__(self, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, nodes):
    super(Cell, self).__init__()
    self.reduction = reduction
    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._nodes = nodes
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    for i in range(self._nodes):
      for j in range(i+2):
        stride = 2 if reduction and j < 2 else 1
        mixed_ops = nn.ModuleList()
        for primitive in PRIMITIVES:
          mixed_ops.append(OPS[primitive](C, stride, affine=False))
        self._ops.append(mixed_ops)
    self._init_alphas()

  def _init_alphas(self):
    k = sum(1 for i in range(self._nodes) for j in range(2+i))
    num_ops = len(PRIMITIVES)

    self._alphas = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)

  def arch_parameters(self):
    return self._alphas

  def forward_hard(self, s0, s1, weights, index):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)
    # print(s0.shape)
    # print(s1.shape)

    states = [s0, s1]
    offset = 0
    for i in range(self._nodes):
      inter_nodes = []
      for j in range(i+2):
          argmaxs = index[offset+j].item()
          weight = weights[offset+j]
          op = self._ops[offset+j]
          h = states[j]
          weightsum = sum(weight[_ie] * op[_ie](h) if _ie == argmaxs else weight[_ie] for _ie, edge in enumerate(op))
          inter_nodes.append(weightsum)
          # print(weightsum.shape)
      s = sum(inter_nodes)
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)

  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._nodes):
      inter_nodes = []
      for j, h in enumerate(states):
        weight = weights[offset+j]
        op = self._ops[offset+j]
        weightsum = sum(weight[ie] * op[ie](h) for ie, edge in enumerate(op))
        inter_nodes.append(weightsum)
      s = sum(inter_nodes)
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)

  def genotype(self):
    def _parse(weights):
      gene = []
      n = 2
      start = 0

      for i in range(self._nodes):
        end = start + n

        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        # edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]

        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
            # if k_best is None or W[j][k] > W[j][k_best]:
              # k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene = _parse(F.softmax(self._alphas, dim=1).data.cpu().numpy())
    concat = range(2 + self._nodes - self._multiplier, self._nodes + 2)
    genotype = Genotype_stage(cell=gene, cell_concat=concat)
    return genotype


if __name__ == "__main__":
  
  tensor = torch.randn(2, 3, 256, 128)
  weights = [1.,1.,1.,1.,1.,1.]
  # print(np.prod(torch.randn(1,1,192, 512).size())/ 1e6)
  # exit(1)
  # model = MBlock(24, 24, usesub = False, usek9 = False)
  # model = Cell(24, 24, usesub = False, usek9 = True)
  model = Network(1000, cfg)
  # print(model)

  # res = model(tensor, weights)
  res = model(tensor)
  print(res[0].shape)
  print(res[1].shape)
  genotypes = model.genotype()
  for i, genotype in enumerate(genotypes):
    print('{}th, {}'.format(i, genotype))
  model._parse_genotype()

