import sys
import os
import copy
import json
import time

# command_list =[
#     "CUDA_VISIBLE_DEVICES=0 nohup make byte_track_nlla det_check_path=check/check_loss_two_step_center_rh_ind mode=upperbound nll_threshold=",
#     "CUDA_VISIBLE_DEVICES=0 nohup make eval mode=upperbound output_name=center_rh_ind_nll_",
# ]

command_list =[
    "CUDA_VISIBLE_DEVICES=0 nohup make byter_track_cfk_nlla split=val det_check_path=check/check_loss_two_step_center_rh_ind_val_cov mode=upperbound nll_threshold=",
    "CUDA_VISIBLE_DEVICES=0 nohup make eval mode=upperbound split=val output_name=center_rh_ind_nll_cfk_",
]

# command_list =[
#     "CUDA_VISIBLE_DEVICES=0 nohup make sort_cfk_nlla split=val det_check_path=check/check_loss_two_step_center_sr_ind_val_cov mode=upperbound nll_threshold=",
#     "CUDA_VISIBLE_DEVICES=0 nohup make eval mode=upperbound split=val output_name=center_sr_ind_nll_cfk_",
# ]


cout_command = " > nll.out 2> nll.err&"

# val_list = [1,2,5,7,10,15,20,30,40,50,60,70,80,90,100,200,300,400,500,1000,2000,3000,5000,10000,20000,40000,100000]
# val_list = [2600,2700,2800,2900,3100,3200,3300,3400,3600,3700,3800,3900,4100,4200,4300,4400,4600,4700,4800,4900]
# val_list = [1,5,10,15,20,30,40,50,60,70,75,80,85,90,95,100,110,120,150,200,300,400]
val_list = [130, 140, 160, 170, 180, 190, 210, 220, 230, 240, 250, 260, 270, 280, 290, 310, 320, 330, 340, 350, 360, 370, 380, 390]
#val_list = [0.5, 1,2,3,4,5,6,7,8,9,10,15,20,30,40,50,60,70,80,90,100,200,300,400,500]
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