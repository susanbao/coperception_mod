import argparse
import os
from copy import deepcopy

import seaborn as sns
import torch.optim as optim
from torch.utils.data import DataLoader

from coperception.datasets import V2XSimDet
from coperception.configs import Config, ConfigGlobal
from coperception.utils.CoDetModule import *
from coperception.utils.loss import *
from coperception.utils.mean_ap import eval_map, eval_nll
from coperception.models.det import *
from coperception.utils.detection_util import late_fusion
from coperception.utils.data_util import apply_pose_noise
import ipdb
import wandb
import socket

def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path

def compute_one_nll(args):
    nepoch = args.nepoch
    data_path = args.resume + "/all_data.npy"
    data = np.load(data_path, allow_pickle=True)
    det_results_all_local = data.item()['det_results_frame']
    annotations_all_local = data.item()['annotations_frame']
    if args.covar_path != "":
        covar_data = np.load(args.covar_path, allow_pickle=True)
        covar_e = covar_data.item()['covar_e']
        covar_a = covar_data.item()['covar_a']
        covar_e = torch.from_numpy(covar_e)
        covar_a = torch.from_numpy(covar_a)
        w = 0.5
    else:
        covar_a = None
        covar_e = None
        w = 0.0
    print(
        "Quantitative evaluation results of model from {}, at epoch {}".format(
            args.resume, nepoch
        )
    )
    print("NLL:")
    covar_nll = eval_nll(det_results_all_local, annotations_all_local, scale_ranges=None, iou_thr=0.0, covar_e = covar_e, covar_a=covar_a, w=w)
    print(covar_nll)

def compute_null_with_different_weight(args):
    nepoch = args.nepoch
    data_path = args.resume + "/all_data.npy"
    data = np.load(data_path, allow_pickle=True)
    det_results_all_local = data.item()['det_results_frame']
    annotations_all_local = data.item()['annotations_frame']
    covar_data = np.load(args.covar_path, allow_pickle=True)
    covar_e = covar_data.item()['covar_e']
    covar_a = covar_data.item()['covar_a']
    covar_e = torch.from_numpy(covar_e)
    covar_a = torch.from_numpy(covar_a)
    
    #save with wandb
    wandb_path = args.resume + "/wandb"
    if not os.path.exists(wandb_path):
        os.makedirs(wandb_path)
    wandb.init(config=args,
            project="mbb_weight",
            entity="susanbao",
            notes=socket.gethostname(),
            name=args.resume,
            dir=wandb_path,
            job_type="testing",
            reinit=True)
    
    w_list = np.arange(0.0, 1.05, 0.05)
    nll_list = []
    for index, w in enumerate(w_list):
        covar_nll = eval_nll(det_results_all_local, annotations_all_local, scale_ranges=None, iou_thr=0.0, covar_e = covar_e, covar_a=covar_a, w=w)
        wandb.log({"NLL": covar_nll[0]['NLL'], "w": w}, step=index)
        nll_list.append(covar_nll[0]['NLL'])
    save_data = {"nll_list": nll_list, "w_list":w_list}
    save_data_path = args.resume + "/nll_list.npy"
    np.save(save_data_path, save_data)
    print("Complete save computed NLLs in {}".format(save_data_path))
    if args.use_wandb:
        wandb.finish()

def compute_nll_only_with_mbb(args):
    covar_data = np.load(args.covar_path, allow_pickle=True)
    covar_e = covar_data.item()['covar_e']
    covar_e = torch.from_numpy(covar_e)
    nepoch = args.nepoch
    data_path = args.resume + "/all_data.npy"
    data = np.load(data_path, allow_pickle=True)
    det_results_all_local = data.item()['det_results_frame']
    annotations_all_local = data.item()['annotations_frame']
    print(
        "Quantitative evaluation results of model from {}, at epoch {}".format(
            args.resume, nepoch
        )
    )
    print("NLL:")
    covar_nll = eval_nll(det_results_all_local, annotations_all_local, scale_ranges=None, iou_thr=0.0, covar_e = covar_e)
    print(covar_nll)

def main(args):
    if args.type == 0:
        compute_one_nll(args)
    elif args.type == 1:
        compute_null_with_different_weight(args)
    else:
        print("Error: type is error!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nepoch", default=100, type=int, help="Number of epochs")
    parser.add_argument("--nworker", default=1, type=int, help="Number of workers")
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="The path of saving the testing results",
    )
    parser.add_argument(
        "--covar_path",
        default="",
        type=str,
        help="The path of saving the computed covariance matrix from mbb",
    )
    parser.add_argument(
        "--type",
        default=0, 
        type=int,
        help="0: compute nll once, 1: compute null for different weight and draw figure",
    )

    torch.multiprocessing.set_sharing_strategy("file_system")
    args = parser.parse_args()
    print(args)
    main(args)