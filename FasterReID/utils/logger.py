# -*- coding: utf-8 -*-
# @Author: solicucu

import logging
import os
import sys 

def setup_logger(name, save_dir, distributed_rank, log_name = "log.txt"):

	logger = logging.getLogger(name) 
	logger.setLevel(logging.DEBUG)

	# do not log the results for the non-master process
	if distributed_rank > 0:
		return logger 
	# redirect the output to the screen
	stdh = logging.StreamHandler(stream = sys.stdout)
	stdh.setLevel(logging.DEBUG)
	formater = logging.Formatter("%(filename)s line %(lineno)s %(asctime)s %(name)s %(levelname)s: %(message)s")

	stdh.setFormatter(formater)
	logger.addHandler(stdh)

	log_dir = save_dir + "logs/"
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)

	# log the result to the text
	if save_dir:
		fh = logging.FileHandler(os.path.join(log_dir, log_name), mode = 'w')
		fh.setLevel(logging.DEBUG)
		fh.setFormatter(formater)
		logger.addHandler(fh)

	return logger 