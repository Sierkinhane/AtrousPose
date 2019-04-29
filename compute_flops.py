import torch
from ptflops import get_model_complexity_info
from model.deeplabv3 import PoseLab_50_2b_v3
with torch.cuda.device(0):
	net = PoseLab_50_2b_v3()
	flops, params = get_model_complexity_info(net, (384,384), as_strings=True, print_per_layer_stat=True)
	print("flops:  " + flops)
	print("params: " + params)