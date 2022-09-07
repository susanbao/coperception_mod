import argparse
import os
from copy import deepcopy

from coperception.utils.CoDetModule import *
from coperception.utils.loss import *
from coperception.utils.mean_ap import eval_map, eval_nll, get_residual_error
import ipdb

def main(args):
    start_epoch = args.min_epoch
    end_epoch = args.max_epoch
    res_diff = []
    for epoch in range(start_epoch, end_epoch+1):
        data_path = args.mbb_path + "/{}".format(epoch) +"/all_data.npy"
        print("Load data from {}".format(data_path))
        data = np.load(data_path, allow_pickle=True)
        det_results_all_local = data.item()['det_results_frame']
        annotations_all_local = data.item()['annotations_frame']
        res_diff_one_epoch = get_residual_error(det_results_all_local, annotations_all_local, scale_ranges=None, iou_thr=0.0)
        res_diff.extend(res_diff_one_epoch)
        print("Number of corners of all bounding box: {}".format(len(res_diff[epoch])))
    res_diff_np = np.array(res_diff[0])
    for i in range(1, len(res_diff)):
        res_diff_np = np.concatenate((res_diff_np, res_diff[i]))
    print(res_diff_np.shape)
    print("covariance matrix:")
    print(np.cov(res_diff_np.T))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_epoch", default=0, type=int, help="min epochs we consider")
    parser.add_argument("--max_epoch", default=25, type=int, help="max epochs we consider")
    parser.add_argument("--nworker", default=1, type=int, help="Number of workers")
    parser.add_argument(
        "--mbb_path",
        default="",
        type=str,
        help="The path to the serval mbb models",
    )

    torch.multiprocessing.set_sharing_strategy("file_system")
    args = parser.parse_args()
    print(args)
    main(args)