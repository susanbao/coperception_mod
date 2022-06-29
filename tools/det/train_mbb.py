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


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path

def test_model(fafmodule, validation_data_loader, flag, device, config, start_epoch, args):
    fafmodule.model.eval()
    num_agent = args.num_agent
    apply_late_fusion = args.apply_late_fusion
    agent_idx_range = range(num_agent) if args.rsu else range(1, num_agent)
    save_fig_path = [
        check_folder(os.path.join(args.test_store, f"vis{i}")) for i in agent_idx_range
    ]
    tracking_path = [
        check_folder(os.path.join(args.test_store, f"tracking{i}"))
        for i in agent_idx_range
    ]

    # for local and global mAP evaluation
    det_results_local = [[] for i in agent_idx_range]
    annotations_local = [[] for i in agent_idx_range]

    tracking_file = [set()] * num_agent
    for cnt, sample in enumerate(validation_data_loader):
        t = time.time()
        (
            padded_voxel_point_list,
            padded_voxel_points_teacher_list,
            label_one_hot_list,
            reg_target_list,
            reg_loss_mask_list,
            anchors_map_list,
            vis_maps_list,
            gt_max_iou,
            filenames,
            target_agent_id_list,
            num_agent_list,
            trans_matrices_list,
        ) = zip(*sample)

        print(filenames)

        filename0 = filenames[0]
        trans_matrices = torch.stack(tuple(trans_matrices_list), 1)
        target_agent_ids = torch.stack(tuple(target_agent_id_list), 1)
        num_all_agents = torch.stack(tuple(num_agent_list), 1)

        # add pose noise
        if args.pose_noise > 0:
            apply_pose_noise(args.pose_noise, trans_matrices)

        if not args.rsu:
            num_all_agents -= 1

        if flag == "upperbound":
            padded_voxel_points = torch.cat(tuple(padded_voxel_points_teacher_list), 0)
        else:
            padded_voxel_points = torch.cat(tuple(padded_voxel_point_list), 0)

        label_one_hot = torch.cat(tuple(label_one_hot_list), 0)
        reg_target = torch.cat(tuple(reg_target_list), 0)
        reg_loss_mask = torch.cat(tuple(reg_loss_mask_list), 0)
        anchors_map = torch.cat(tuple(anchors_map_list), 0)
        vis_maps = torch.cat(tuple(vis_maps_list), 0)

        data = {
            "bev_seq": padded_voxel_points.to(device),
            "labels": label_one_hot.to(device),
            "reg_targets": reg_target.to(device),
            "anchors": anchors_map.to(device),
            "vis_maps": vis_maps.to(device),
            "reg_loss_mask": reg_loss_mask.to(device).type(dtype=torch.bool),
            "target_agent_ids": target_agent_ids.to(device),
            "num_agent": num_all_agents.to(device),
            "trans_matrices": trans_matrices.to(device),
        }

        if flag == "lowerbound_box_com":
            loss, cls_loss, loc_loss, result = fafmodule.predict_all_with_box_com(
                data, data["trans_matrices"]
            )
        elif flag == "disco":
            (
                loss,
                cls_loss,
                loc_loss,
                result,
                save_agent_weight_list,
            ) = fafmodule.predict_all(data, 1, num_agent=num_agent)
        else:
            loss, cls_loss, loc_loss, result = fafmodule.predict_all(
                data, 1, num_agent=num_agent
            )

        box_color_map = ["red", "yellow", "blue", "purple", "black", "orange"]

        # If has RSU, do not count RSU's output into evaluation
        eval_start_idx = 1 if args.rsu else 0

        # local qualitative evaluation
        for k in range(eval_start_idx, num_agent):
            box_colors = None
            if apply_late_fusion == 1 and len(result[k]) != 0:
                pred_restore = result[k][0][0][0]["pred"]
                score_restore = result[k][0][0][0]["score"]
                selected_idx_restore = result[k][0][0][0]["selected_idx"]

            data_agents = {
                "bev_seq": torch.unsqueeze(padded_voxel_points[k, :, :, :, :], 1),
                "reg_targets": torch.unsqueeze(reg_target[k, :, :, :, :, :], 0),
                "anchors": torch.unsqueeze(anchors_map[k, :, :, :, :], 0),
            }
            temp = gt_max_iou[k]

            if len(temp[0]["gt_box"]) == 0:
                data_agents["gt_max_iou"] = []
            else:
                data_agents["gt_max_iou"] = temp[0]["gt_box"][0, :, :]

            # late fusion
            if apply_late_fusion == 1 and len(result[k]) != 0:
                box_colors = late_fusion(
                    k, num_agent, result, trans_matrices, box_color_map
                )

            result_temp = result[k]

            temp = {
                "bev_seq": data_agents["bev_seq"][0, -1].cpu().numpy(),
                "result": [] if len(result_temp) == 0 else result_temp[0][0],
                "reg_targets": data_agents["reg_targets"].cpu().numpy()[0],
                "anchors_map": data_agents["anchors"].cpu().numpy()[0],
                "gt_max_iou": data_agents["gt_max_iou"],
            }
            det_results_local[k], annotations_local[k] = cal_local_mAP(
                config, temp, det_results_local[k], annotations_local[k]
            )

            filename = str(filename0[0][0])
            cut = filename[filename.rfind("agent") + 7 :]
            seq_name = cut[: cut.rfind("_")]
            idx = cut[cut.rfind("_") + 1 : cut.rfind("/")]
            seq_save = os.path.join(save_fig_path[k], seq_name)
            check_folder(seq_save)
            idx_save = str(idx) + ".png"
            temp_ = deepcopy(temp)
            if args.visualization:
                visualization(
                    config,
                    temp,
                    box_colors,
                    box_color_map,
                    apply_late_fusion,
                    os.path.join(seq_save, idx_save),
                )

            # == tracking ==
            if args.tracking:
                scene, frame = filename.split("/")[-2].split("_")
                det_file = os.path.join(tracking_path[k], f"det_{scene}.txt")
                if scene not in tracking_file[k]:
                    det_file = open(det_file, "w")
                    tracking_file[k].add(scene)
                else:
                    det_file = open(det_file, "a")
                det_corners = get_det_corners(config, temp_)
                for ic, c in enumerate(det_corners):
                    det_file.write(
                        ",".join(
                            [
                                str(
                                    int(frame) + 1
                                ),  # frame idx is 1-based for tracking
                                "-1",
                                "{:.2f}".format(c[0]),
                                "{:.2f}".format(c[1]),
                                "{:.2f}".format(c[2]),
                                "{:.2f}".format(c[3]),
                                str(result_temp[0][0][0]["score"][ic]),
                                "-1",
                                "-1",
                                "-1",
                            ]
                        )
                        + "\n"
                    )
                    det_file.flush()

                det_file.close()

            # restore data before late-fusion
            if apply_late_fusion == 1 and len(result[k]) != 0:
                result[k][0][0][0]["pred"] = pred_restore
                result[k][0][0][0]["score"] = score_restore
                result[k][0][0][0]["selected_idx"] = selected_idx_restore

        print("Validation scene {}, at frame {}".format(seq_name, idx))
        print("Takes {} s\n".format(str(time.time() - t)))

    logger_root = args.logpath if args.logpath != "" else "logs"
    logger_root = os.path.join(
        logger_root, f"{flag}_eval", "with_rsu" if args.rsu else "no_rsu"
    )
    os.makedirs(logger_root, exist_ok=True)
    log_file_path = os.path.join(logger_root, "log_test.txt")
    log_file = open(log_file_path, "w")

    def print_and_write_log(log_str):
        print(log_str)
        log_file.write(log_str + "\n")

    mean_ap_local = []
    # local mAP evaluation
    det_results_all_local = []
    annotations_all_local = []
    for k in range(eval_start_idx, num_agent):
        if type(det_results_local[k]) != list or len(det_results_local[k]) == 0:
            continue

        print_and_write_log("Local mAP@0.5 from agent {}".format(k))
        mean_ap, _ = eval_map(
            det_results_local[k],
            annotations_local[k],
            scale_ranges=None,
            iou_thr=0.5,
            dataset=None,
            logger=None,
        )
        mean_ap_local.append(mean_ap)
        print_and_write_log("Local mAP@0.7 from agent {}".format(k))

        ean_ap, _ = eval_map(
            det_results_local[k],
            annotations_local[k],
            scale_ranges=None,
            iou_thr=0.7,
            dataset=None,
            logger=None,
        )
        mean_ap_local.append(mean_ap)

        det_results_all_local += det_results_local[k]
        annotations_all_local += annotations_local[k]

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

    print_and_write_log(
        "Quantitative evaluation results of model from {}, at epoch {}".format(
            args.resume, start_epoch - 1
        )
    )

    for k in range(eval_start_idx, num_agent):
        print_and_write_log(
            "agent{} mAP@0.5 is {} and mAP@0.7 is {}".format(
                k + 1 if not args.rsu else k, mean_ap_local[k * 2], mean_ap_local[(k * 2) + 1]
            )
        )

    print_and_write_log(
        "average local mAP@0.5 is {} and average local mAP@0.7 is {}".format(
            mean_ap_local[-2], mean_ap_local[-1]
        )
    )

    #if need_log:
    #   saver.close()

def main(args):
    config = Config("train", binary=True, only_det=True)
    config_global = ConfigGlobal("train", binary=True, only_det=True)

    num_epochs = args.nepoch
    need_log = args.log
    num_workers = args.nworker
    start_epoch = 1
    batch_size = args.batch_size
    compress_level = args.compress_level
    auto_resume_path = args.auto_resume_path
    pose_noise = args.pose_noise
    only_v2i = args.only_v2i

    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)

    if args.com == "upperbound":
        flag = "upperbound"
    elif args.com == "when2com" and args.warp_flag:
        flag = "when2com_warp"
    elif args.com in [
        "lowerbound",
        "v2v",
        "disco",
        "sum",
        "mean",
        "max",
        "cat",
        "agent",
        "when2com",
    ]:
        flag = args.com
    else:
        raise ValueError(f"com: {args.com} is not supported")

    config.flag = flag

    num_agent = args.num_agent
    # agent0 is the RSU
    agent_idx_range = range(num_agent) if args.rsu else range(1, num_agent)
    training_dataset = V2XSimDet(
        dataset_roots=[f"{args.data}/agent{i}" for i in agent_idx_range],
        config=config,
        config_global=config_global,
        split="train",
        bound="upperbound" if args.com == "upperbound" else "lowerbound",
        kd_flag=args.kd_flag,
        rsu=args.rsu,
    )
    training_sample = MbbSampler(V2XSimDet, args.block_len)
    training_data_loader = DataLoader(
        training_dataset, sample=training_sample, batch_size=batch_size, num_workers=num_workers
    )
    print("Training dataset size:", len(training_dataset))
    if args.test:
        validation_dataset = V2XSimDet(
            dataset_roots=[f"{args.test_data}/agent{i}" for i in agent_idx_range],
            config=config,
            config_global=config_global,
            split="val",
            val=True,
            bound="upperbound" if args.com == "upperbound" else "lowerbound",
            kd_flag=args.kd_flag,
            rsu=args.rsu,
        )
        validation_data_loader = DataLoader(
            validation_dataset, batch_size=1, shuffle=False, num_workers=num_workers
        )
        print("Validation dataset size:", len(validation_dataset))
    else:
        validation_dataset = None
        validation_data_loader = None

    logger_root = args.logpath if args.logpath != "" else "logs"

    if not args.rsu:
        num_agent -= 1

    if flag == "lowerbound" or flag == "upperbound":
        model = FaFNet(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
        )
    elif flag == "when2com" or flag == "when2com_warp":
        model = When2com(
            config,
            layer=args.layer,
            warp_flag=args.warp_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif flag == "v2v":
        model = V2VNet(
            config,
            gnn_iter_times=args.gnn_iter_times,
            layer=args.layer,
            layer_channel=256,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif flag == "disco":
        model = DiscoNet(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif flag == "sum":
        model = SumFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif flag == "mean":
        model = MeanFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif flag == "max":
        model = MaxFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif flag == "cat":
        model = CatFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif flag == "agent":
        model = AgentWiseWeightedFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )

    model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = {
        "cls": SoftmaxFocalClassificationLoss(),
        "loc": WeightedSmoothL1LocalizationLoss(),
    }

    if args.kd_flag == 1:
        teacher = TeacherNet(config)
        teacher = nn.DataParallel(teacher)
        teacher = teacher.to(device)
        faf_module = FaFModule(
            model, teacher, config, optimizer, criterion, args.kd_flag
        )
        checkpoint_teacher = torch.load(args.resume_teacher)
        start_epoch_teacher = checkpoint_teacher["epoch"]
        faf_module.teacher.load_state_dict(checkpoint_teacher["model_state_dict"])
        print(
            "Load teacher model from {}, at epoch {}".format(
                args.resume_teacher, start_epoch_teacher
            )
        )
        faf_module.teacher.eval()
    else:
        faf_module = FaFModule(model, model, config, optimizer, criterion, args.kd_flag)

    rsu_path = "with_rsu" if args.rsu else "no_rsu"
    model_save_path = check_folder(logger_root)
    model_save_path = check_folder(os.path.join(model_save_path, flag))

    if args.rsu:
        model_save_path = check_folder(os.path.join(model_save_path, "with_rsu"))
    else:
        model_save_path = check_folder(os.path.join(model_save_path, "no_rsu"))

    # check if there is valid check point file
    has_valid_pth = False
    for pth_file in os.listdir(os.path.join(auto_resume_path, f"{flag}/{rsu_path}")):
        if pth_file.startswith("epoch_") and pth_file.endswith(".pth"):
            has_valid_pth = True
            break

    if not has_valid_pth:
        if args.init_resume_path == "":
            print(
                f"No valid check point file in {auto_resume_path} and {args.init_resume_path} dir, weights not loaded."
            )
        auto_resume_path = ""

    if args.resume == "" and auto_resume_path == "":
        log_file_name = os.path.join(model_save_path, "log.txt")
        saver = open(log_file_name, "w")
        saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
        saver.flush()

        # Logging the details for this experiment
        saver.write("command line: {}\n".format(" ".join(sys.argv[0:])))
        saver.write(args.__repr__() + "\n\n")
        saver.flush()
    else:
        if auto_resume_path != "":
            model_save_path = os.path.join(auto_resume_path, f"{flag}/{rsu_path}")
        else:
            model_save_path = args.resume[: args.resume.rfind("/")]

        print(f"model save path: {model_save_path}")

        log_file_name = os.path.join(model_save_path, "log.txt")

        if os.path.exists(log_file_name):
            saver = open(log_file_name, "a")
        else:
            os.makedirs(model_save_path, exist_ok=True)
            saver = open(log_file_name, "w")

        saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
        saver.flush()

        # Logging the details for this experiment
        saver.write("command line: {}\n".format(" ".join(sys.argv[1:])))
        saver.write(args.__repr__() + "\n\n")
        saver.flush()
        if args.init_resume_path != '':
            checkpoint = torch.load(args.init_resume_path)
            print("Load pth from init path {args.init_resume_path}")
        elif auto_resume_path != "":
            list_of_files = glob.glob(f"{model_save_path}/*.pth")
            latest_pth = max(list_of_files, key=os.path.getctime)
            checkpoint = torch.load(latest_pth)
            print("Load pth from auto resume path {auto_resume_path}")
        else:
            checkpoint = torch.load(args.resume)
            print("Load pth from resume path {args.resume}")

        start_epoch = checkpoint["epoch"] + 1
        faf_module.model.load_state_dict(checkpoint["model_state_dict"])
        faf_module.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        faf_module.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print("Load model from {}, at epoch {}".format(args.resume, start_epoch - 1))

    for epoch in range(start_epoch, num_epochs + 1):
        lr = faf_module.optimizer.param_groups[0]["lr"]
        print("Epoch {}, learning rate {}".format(epoch, lr))

        if need_log:
            saver.write("epoch: {}, lr: {}\t".format(epoch, lr))
            saver.flush()

        running_loss_disp = AverageMeter("Total loss", ":.6f")
        running_loss_class = AverageMeter(
            "classification Loss", ":.6f"
        )  # for cell classification error
        running_loss_loc = AverageMeter(
            "Localization Loss", ":.6f"
        )  # for state estimation error

        faf_module.model.train()

        t = tqdm(training_data_loader)
        for sample in t:
            (
                padded_voxel_point_list,
                padded_voxel_points_teacher_list,
                label_one_hot_list,
                reg_target_list,
                reg_loss_mask_list,
                anchors_map_list,
                vis_maps_list,
                target_agent_id_list,
                num_agent_list,
                trans_matrices_list,
            ) = zip(*sample)

            trans_matrices = torch.stack(tuple(trans_matrices_list), 1)
            target_agent_id = torch.stack(tuple(target_agent_id_list), 1)
            num_all_agents = torch.stack(tuple(num_agent_list), 1)

            # add pose noise
            if pose_noise > 0:
                apply_pose_noise(pose_noise, trans_matrices)

            if not args.rsu:
                num_all_agents -= 1

            if flag == "upperbound":
                padded_voxel_point = torch.cat(
                    tuple(padded_voxel_points_teacher_list), 0
                )
            else:
                padded_voxel_point = torch.cat(tuple(padded_voxel_point_list), 0)

            label_one_hot = torch.cat(tuple(label_one_hot_list), 0)
            reg_target = torch.cat(tuple(reg_target_list), 0)
            reg_loss_mask = torch.cat(tuple(reg_loss_mask_list), 0)
            anchors_map = torch.cat(tuple(anchors_map_list), 0)
            vis_maps = torch.cat(tuple(vis_maps_list), 0)

            data = {
                "bev_seq": padded_voxel_point.to(device),
                "labels": label_one_hot.to(device),
                "reg_targets": reg_target.to(device),
                "anchors": anchors_map.to(device),
                "reg_loss_mask": reg_loss_mask.to(device).type(dtype=torch.bool),
                "vis_maps": vis_maps.to(device),
                "target_agent_ids": target_agent_id.to(device),
                "num_agent": num_all_agents.to(device),
                "trans_matrices": trans_matrices,
            }

            if args.kd_flag == 1:
                padded_voxel_points_teacher = torch.cat(
                    tuple(padded_voxel_points_teacher_list), 0
                )
                data["bev_seq_teacher"] = padded_voxel_points_teacher.to(device)
                data["kd_weight"] = args.kd_weight

            loss, cls_loss, loc_loss = faf_module.step(
                data, batch_size, num_agent=num_agent
            )
            running_loss_disp.update(loss)
            running_loss_class.update(cls_loss)
            running_loss_loc.update(loc_loss)

            if np.isnan(loss) or np.isnan(cls_loss) or np.isnan(loc_loss):
                print(f"Epoch {epoch}, loss is nan: {loss}, {cls_loss} {loc_loss}")
                sys.exit()

            t.set_description("Epoch {},     lr {}".format(epoch, lr))
            t.set_postfix(
                cls_loss=running_loss_class.avg, loc_loss=running_loss_loc.avg
            )

        faf_module.scheduler.step()

        # save model
        if need_log:
            saver.write(
                "{}\t{}\t{}\n".format(
                    running_loss_disp, running_loss_class, running_loss_loc
                )
            )
            saver.flush()
            if config.MGDA:
                save_dict = {
                    "epoch": epoch,
                    "encoder_state_dict": faf_module.encoder.state_dict(),
                    "optimizer_encoder_state_dict": faf_module.optimizer_encoder.state_dict(),
                    "scheduler_encoder_state_dict": faf_module.scheduler_encoder.state_dict(),
                    "head_state_dict": faf_module.head.state_dict(),
                    "optimizer_head_state_dict": faf_module.optimizer_head.state_dict(),
                    "scheduler_head_state_dict": faf_module.scheduler_head.state_dict(),
                    "loss": running_loss_disp.avg,
                }
            else:
                save_dict = {
                    "epoch": epoch,
                    "model_state_dict": faf_module.model.state_dict(),
                    "optimizer_state_dict": faf_module.optimizer.state_dict(),
                    "scheduler_state_dict": faf_module.scheduler.state_dict(),
                    "loss": running_loss_disp.avg,
                }
            torch.save(
                save_dict, os.path.join(model_save_path, "epoch_" + str(epoch) + ".pth")
            )

    if need_log:
        saver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        default=None,
        type=str,
        help="The path to the preprocessed sparse BEV training data",
    )
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
    parser.add_argument("--nepoch", default=100, type=int, help="Number of epochs/boostraps")
    parser.add_argument("--nworker", default=2, type=int, help="Number of workers")
    parser.add_argument("--lr", default=0.001, type=float, help="Initial learning rate")
    parser.add_argument("--log", action="store_true", help="Whether to log")
    parser.add_argument("--logpath", default="", help="The path to the output log file")
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="The path to the saved model that is loaded to resume training",
    )
    parser.add_argument(
        "--resume_teacher",
        default="",
        type=str,
        help="The path to the saved teacher model that is loaded to resume training",
    )
    parser.add_argument(
        "--layer",
        default=3,
        type=int,
        help="Communicate which layer in the single layer com mode",
    )
    parser.add_argument(
        "--warp_flag", default=0, type=int, help="Whether to use pose info for When2com"
    )
    parser.add_argument(
        "--kd_flag",
        default=0,
        type=int,
        help="Whether to enable distillation (only DiscNet is 1 )",
    )
    parser.add_argument("--kd_weight", default=100000, type=int, help="KD loss weight")
    parser.add_argument(
        "--gnn_iter_times",
        default=3,
        type=int,
        help="Number of message passing for V2VNet",
    )
    parser.add_argument(
        "--visualization", default=True, help="Visualize validation result"
    )
    parser.add_argument(
        "--com",
        default="",
        type=str,
        help="lowerbound/upperbound/disco/when2com/v2v/sum/mean/max/cat/agent",
    )
    parser.add_argument("--rsu", default=0, type=int, help="0: no RSU, 1: RSU")
    parser.add_argument(
        "--num_agent", default=6, type=int, help="The total number of agents"
    )
    parser.add_argument(
        "--auto_resume_path",
        default="",
        type=str,
        help="The path to automatically reload the latest pth",
    )
    parser.add_argument(
        "--compress_level",
        default=0,
        type=int,
        help="Compress the communication layer channels by 2**x times in encoder",
    )
    parser.add_argument(
        "--pose_noise",
        default=0,
        type=float,
        help="draw noise from normal distribution with given mean (in meters), apply to transformation matrix.",
    )
    parser.add_argument(
        "--only_v2i",
        default=0,
        type=int,
        help="1: only v2i, 0: v2v and v2i",
    )
    parser.add_argument(
        "--test", action="store_true", help="Whether to test on the test dataset and store the result",
    )
    parser.add_argument(
        "--test_data",
        default="",
        type=str,
        help="The path to the preprocessed sparse BEV testing data",
    )
    parser.add_argument(
        "--test_store",
        default="",
        type=str,
        help="The path to store the output of testing",
    )
    parser.add_argument(
        "--apply_late_fusion",
        default=0,
        type=int,
        help="1: apply late fusion. 0: no late fusion",
    )
    parser.add_argument(
        "--init_resume_path",
        default="",
        type=str,
        help="The path to reload the initial pth",
    )
    parser.add_argument(
        "--block_len",
        default=16,
        type=int,
        help="Length of block in MBB method",
    )

    torch.multiprocessing.set_sharing_strategy("file_system")
    args = parser.parse_args()
    print(args)
    main(args)
