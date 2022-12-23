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
from coperception.utils.mean_ap import match_pairs
from coperception.models.det import *
import ipdb

def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path
    
def main(args):
    data_path = args.resume
    iou = args.iou
    data = np.load(data_path, allow_pickle=True)
    file_name = data_path.split('/')[-1]
    file_name = file_name.split('.')[0]
    file_name = "match_" + file_name + "_" + str(iou) + ".npy"
    iou_float = iou / 10
    det_results_all_local = data.item()['det_results_frame']
    annotations_all_local = data.item()['annotations_frame']
    tp_results, fp_results = match_pairs(det_results_all_local, annotations_all_local, scale_ranges=None, iou_thr=iou_float)
    output_file = os.path.join(args.save_path, file_name)
    match_results = {"tp":tp_results, "fp":fp_results}
    np.save(output_file, match_results)
    print("Complete pair matching on IOU={} and store in file {}".format(iou_float, output_file))
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="The path of saving the testing results",
    )
    parser.add_argument(
        "--save_path",
        default="",
        type=str,
        help="The path to save the result",
    )
    parser.add_argument(
        "--exp_name",
        default="",
        type=str,
        help="exp name",
    )
    parser.add_argument(
        "--iou",
        default=5, 
        type=int,
        help="iou threshold 5 means 0.5, 7 means 0.7",
    )
    torch.multiprocessing.set_sharing_strategy("file_system")
    args = parser.parse_args()
    print(args)
    main(args)