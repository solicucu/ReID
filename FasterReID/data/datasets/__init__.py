# -*- coding: utf-8 -*-
# @Author: solicucu

from .market1501 import Market1501 
from .dukemtcm import DukeMTMC
from .msmt17 import MSMT17 
from .imageDataset import ImageDataset 

datasets = {
	
	'market1501': Market1501,
	'dukemtmc': DukeMTMC,
	'msmt17': MSMT17
}

def get_dataset_names():

	return datasets.keys()


def init_dataset(name, *args, **kwargs):

	if name not in datasets.keys():
		raise KeyError("Unknow dataset：{}".format(name))

	return datasets[name](*args, **kwargs)
