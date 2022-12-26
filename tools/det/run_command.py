import sys
import os
import copy
import json
import time

command_list =[
    "CUDA_VISIBLE_DEVICES=3 nohup make test_no_rsu_val  nepoch=100 com=upperbound logpath=check/check_loss_two_step_corner_val loss_type=kl_loss_corner > err_logs/val_upperbound_two_step_corner_ind.out 2> err_logs/val_upperbound_two_step_corner_ind.err&",
    "CUDA_VISIBLE_DEVICES=3 nohup make test_no_rsu  nepoch=100 com=disco logpath=check/check_loss_two_step_corner loss_type=kl_loss_corner > err_logs/test_disco_two_step_corner_ind.out 2> err_logs/test_disco_two_step_corner_ind.err&",
    "CUDA_VISIBLE_DEVICES=3 nohup make test_no_rsu  nepoch=100 com=upperbound logpath=check/check_loss_two_step_corner loss_type=kl_loss_corner > err_logs/test_upperbound_two_step_corner_ind.out 2> err_logs/test_upperbound_two_step_corner_ind.err&"
]

for command in command_list:
	print(command)
	os.system(command)
	print("------------------------sleep------------------------")
	time.sleep(7200) # sleep for 1hours