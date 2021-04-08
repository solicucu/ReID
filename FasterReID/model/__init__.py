# -*- coding: utf-8 -*-
# @Author: solicucu
from .backbone import CDNetwork
from .backbone import CNetwork 
from .models import BaseNet 

def build_model(cfg, num_class):

	model = BaseNet(num_class, cfg) 

	return model 