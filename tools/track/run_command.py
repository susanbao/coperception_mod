import sys
import os
import copy
import json
import time

command_list =[
    "CUDA_VISIBLE_DEVICES=0 nohup make sort_uq det_check_path=check/check_loss_two_step_center_sr_ind mode=upperbound var_threshold=",
    "CUDA_VISIBLE_DEVICES=0 nohup make eval mode=upperbound output_name=center_sr_ind_var_",
]

cout_command = " > nll.out 2> nll.err&"

#val_list = [1,5,10,15,20,30,40,50,60,70,80,90,100,200,300,400,500,1000,2000,3000,5000,10000,20000,40000,100000]
val_list = [0.5, 1,2,3,4,5,6,7,8,9,10,15,20,30,40,50,60,70,80,90,100,200,300,400,500]
for val in val_list:
    c1 = command_list[0] + str(val) + cout_command
    print(c1)
    os.system(c1)
    time.sleep(10)
    c2 = command_list[1] + str(val) + cout_command
    print(c2)
    os.system(c2)
    time.sleep(10)
    print("------------------------sleep------------------------")