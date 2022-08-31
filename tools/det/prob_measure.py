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

def main(args):
    nepoch = args.nepoch
    data = np.load(args.resume, allow_pickle=True)
    det_results_all_local = data.item()['det_results_frame']
    annotations_all_local = data.item()['annotations_frame']
    print(
        "Quantitative evaluation results of model from {}, at epoch {}".format(
            args.resume, nepoch
        )
    )
    print("Average measure with IOU threshold=0.5: ")
    mean_ap_local_average_5, _ = eval_map(
        det_results_all_local,
        annotations_all_local,
        scale_ranges=None,
        iou_thr=0.5,
        dataset=None,
        logger=None,
    )
    print("mAP: {}".format(mean_ap_local_average_5))
    #ipdb.set_trace()
    covar_nll_5 = eval_nll(det_results_all_local, annotations_all_local, scale_ranges=None, iou_thr=0.5)
    print(covar_nll_5)
    print("Average measure with IOU threshold=0.7: ")
    mean_ap_local_average_7, _ = eval_map(
        det_results_all_local,
        annotations_all_local,
        scale_ranges=None,
        iou_thr=0.7,
        dataset=None,
        logger=None,
    )
    print("mAP: {}".format(mean_ap_local_average_7))
    covar_nll_7 = eval_nll(det_results_all_local, annotations_all_local, scale_ranges=None, iou_thr=0.7)
    print(covar_nll_7)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nepoch", default=100, type=int, help="Number of epochs")
    parser.add_argument("--nworker", default=1, type=int, help="Number of workers")
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="The path to the saved model that is loaded to resume training",
    )

    torch.multiprocessing.set_sharing_strategy("file_system")
    args = parser.parse_args()
    print(args)
    main(args)