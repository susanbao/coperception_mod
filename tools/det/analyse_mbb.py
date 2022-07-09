import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from copy import deepcopy

from coperception.datasets import V2XSimDet, MbbSampler
from coperception.configs import Config, ConfigGlobal
from coperception.utils.CoDetModule import *
from coperception.utils.loss import *
from coperception.models.det import *
from coperception.utils import AverageMeter
from coperception.utils.data_util import apply_pose_noise
from coperception.utils.mean_ap import eval_map

import glob
import os

def analyse_one_frame_data(file_path, scene， args):
    det_results_all_local = []
    annotations_all_local = []
    mean_ap_local = []
    eval_start_idx = 1
    num_agent = 6
    for k in range(eval_start_idx, num_agent):
        os.makedirs(logger_root, exist_ok=True)
        log_file_path = os.path.join(args.save_path, scene + "_log_test.txt")
        log_file = open(log_file_path, "w")
        def print_and_write_log(log_str):
            print(log_str)
            log_file.write(log_str + "\n")
        file_path_agent = file_path+"/result"+str(k) + "/" + scene + ".npy"
        #print(file_path_agent)
        data = np.load(file_path_agent, allow_pickle=True)
        det_results_local = data.item()['det_results_local']
        annotations_local = data.item()['annotations_local']
        print_and_write_log("Local mAP@0.5 from agent {}".format(k))
        mean_ap, _ = eval_map(
            det_results_local,
            annotations_local,
            scale_ranges=None,
            iou_thr=0.5,
            dataset=None,
            logger=None,
        )
        mean_ap_local.append(mean_ap)
        print_and_write_log("Local mAP@0.7 from agent {}".format(k))

        mean_ap, _ = eval_map(
            det_results_local,
            annotations_local,
            scale_ranges=None,
            iou_thr=0.7,
            dataset=None,
            logger=None,
        )
        mean_ap_local.append(mean_ap)

        det_results_all_local += det_results_local
        annotations_all_local += annotations_local
    mean_ap_local_average, _ = eval_map(
        det_results_all_local,
        annotations_all_local,
        scale_ranges=None,
        iou_thr=0.5,
        dataset=None,
        logger=None,
    )
    mean_ap_local.append(mean_ap_local_average)

    mean_ap_local_average, _ = eval_map(
        det_results_all_local,
        annotations_all_local,
        scale_ranges=None,
        iou_thr=0.7,
        dataset=None,
        logger=None,
    )
    mean_ap_local.append(mean_ap_local_average)
    
    for k in range(eval_start_idx, num_agent):
        print_and_write_log(
            "agent{} mAP@0.5 is {} and mAP@0.7 is {}".format(
                k, mean_ap_local[(k-1) * 2], mean_ap_local[((k-1) * 2) + 1]
            )
        )

    print_and_write_log(
        "average local mAP@0.5 is {} and average local mAP@0.7 is {}".format(
            mean_ap_local[-2], mean_ap_local[-1]
        )
    )
    
def analysis_all_data(args):
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="", type=str, help="the path of saving results")
    parser.add_argument("--save_path", default="", type=str, help="path to output the log file")
    parser.add_argument("--analyse_all", action="store_true", help="whether to analyse all data")
    parser.add_argument(
        "--num_agent", default=6, type=int, help="The total number of agents"
    )
    parser.add_argument("--rsu", default=0, type=int, help="0: no RSU, 1: RSU")
    parser.add_argument(
        "--com",
        default="",
        type=str,
        help="lowerbound/upperbound/disco/when2com/v2v/sum/mean/max/cat/agent",
    )
    torch.multiprocessing.set_sharing_strategy("file_system")
    args = parser.parse_args()
    print(args)
    if args.analyse_all:
        analysis_all_data(args)