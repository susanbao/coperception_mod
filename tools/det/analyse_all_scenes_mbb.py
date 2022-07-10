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

def analyse_one_frame_data(file_path, scene, args):
    det_results_all_local = []
    annotations_all_local = []
    mean_ap_local = []
    eval_start_idx = 1
    num_agent = 6
    os.makedirs(args.save_path, exist_ok=True)
    log_file_path = os.path.join(args.save_path, scene + "_log_test.txt")
    log_file = open(log_file_path, "w")
    def print_and_write_log(log_str):
        print(log_str)
        log_file.write(log_str + "\n")

    for k in range(eval_start_idx, num_agent):
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

def get_all_scene_file_list(files_path):
    fileList = os.listdir(files_path)
    scene_list = set()
    print(files_path)
    for fname in fileList:
        if fname[-4:] != '.npy' or (not fname[-5].isdigit()):
            continue
        scene, frame = fname.split(".")[0].split("_")
        scene_list.add(scene)
    scene_list = list(scene_list)
    scene_dict = {}
    for i in scene_list:
        scene_dict[i] = []
    for fname in fileList:
        if fname[-4:] != '.npy' or (not fname[-5].isdigit()):
            continue
        scene, frame = fname.split(".")[0].split("_")
        scene_dict[scene].append(fname)
    return scene_dict
    
def analysis_all_data(args):
    # generate mean_ap of each agent and all agents in one scene and store in mean_ap_all_scenes_dic's value which will be epoch*2*(num_agent+1)
    # store in all_scenes_mean_ap.npy
    agent_idx_range = [i for i in range(1-args.rsu, args.num_agent)]
    num_agent = args.num_agent if args.rsu else args.num_agent-1
    files_path = args.path
    scene_dict = get_all_scene_file_list(files_path+"/0/result1")
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    log_file_path = os.path.join(args.save_path, "scene_log_test.txt")
    log_file = open(log_file_path, "w")
    def print_and_write_log(log_str):
        print(log_str)
        log_file.write(log_str + "\n")
    mean_ap_all_scenes_dic = {}
    for scene, files in scene_dict.items():
        mean_ap_scenes = []
        print_and_write_log("scene: " + scene)
        for epoch in range(args.nepoch+1):
            mean_ap_5 = []
            mean_ap_7 = []
            det_results_all_local = []
            annotations_all_local = []
            epoch_path = os.path.join(files_path, f"{epoch}")
            agents_data_path = [os.path.join(epoch_path, f"result{i}") for i in agent_idx_range]
            print_and_write_log("Epoch {}".format(epoch))
            for idx, agent in enumerate(agent_idx_range):
                det_results_local = []
                annotations_local = []
                for file in files:
                    file_path = os.path.join(agents_data_path[idx], file)
                    data = np.load(file_path, allow_pickle=True)
                    det_results_local.extend(data.item()["det_results_frame"])
                    annotations_local.extend(data.item()["annotations_frame"])
                det_results_all_local.extend(det_results_local)
                annotations_all_local.extend(annotations_local)
                if len(det_results_local) == 0:
                    mean_ap_5.append(0) # 0 means there is no data for this agent 
                    mean_ap_7.append(0)
                    continue
                print_and_write_log("Local mAP@0.5 from agent {}".format(agent))
                mean_ap, _ = eval_map(
                    det_results_local,
                    annotations_local,
                    scale_ranges=None,
                    iou_thr=0.5,
                    dataset=None,
                    logger=None,
                )
                mean_ap_5.append(mean_ap)
                print_and_write_log("Local mAP@0.7 from agent {}".format(agent))

                mean_ap, _ = eval_map(
                    det_results_local,
                    annotations_local,
                    scale_ranges=None,
                    iou_thr=0.7,
                    dataset=None,
                    logger=None,
                )
                mean_ap_7.append(mean_ap)
            mean_ap_local_average, _ = eval_map(
                det_results_all_local,
                annotations_all_local,
                scale_ranges=None,
                iou_thr=0.5,
                dataset=None,
                logger=None,
            )
            mean_ap_5.append(mean_ap_local_average)

            mean_ap_local_average, _ = eval_map(
                det_results_all_local,
                annotations_all_local,
                scale_ranges=None,
                iou_thr=0.7,
                dataset=None,
                logger=None,
            )
            mean_ap_7.append(mean_ap_local_average)
            mean_ap_scenes.append([mean_ap_5, mean_ap_7])
            for idx, agent in enumerate(agent_idx_range):
                print_and_write_log(
                    "scene {}: agent{} mAP@0.5 is {} and mAP@0.7 is {}".format(
                        scene, agent, mean_ap_scenes[-1][0][idx], mean_ap_scenes[-1][1][idx]
                    )
                )

            print_and_write_log(
                "scene {}: average local mAP@0.5 is {} and average local mAP@0.7 is {}".format(
                    scene, mean_ap_scenes[-1][0][-1], mean_ap_scenes[-1][1][-1]
                )
            )
        mean_ap_all_scenes_dic[scene] = mean_ap_scenes
    npy_frame_file = os.path.join(files_path, "all_scenes_mean_ap.npy")
    np.save(npy_frame_file, mean_ap_all_scenes_dic)

    return
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="", type=str, help="the path of saving results")
    parser.add_argument("--save_path", default="", type=str, help="path to output the log file")
    parser.add_argument("--analyse_all", action="store_true", help="whether to analyse all data")
    parser.add_argument("--nepoch", default=100, type=int, help="Number of epochs/boostraps")
    parser.add_argument(
        "--num_agent", default=6, type=int, help="The total number of agents"
    )
    parser.add_argument("--rsu", default=0, type=int, help="0: no RSU, 1: RSU")
    args = parser.parse_args()
    print(args)
    if args.analyse_all:
        analysis_all_data(args)
