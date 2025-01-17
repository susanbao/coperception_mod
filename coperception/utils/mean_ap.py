from multiprocessing import Pool
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable
from coperception.utils.postprocess import *
import ipdb
from scipy.stats import multivariate_normal

def compute_reg_nll(dets, gt, covar_e = None, covar_a = None, w=0.0):
    assert len(dets) == len(gt)
    dets_loc = torch.from_numpy(dets[:,:8])
    dets_loc = torch.reshape(dets_loc, (-1, 2))
    gt_loc = torch.from_numpy(gt)
    gt_loc = torch.reshape(gt_loc, (-1, 2))
    if len(dets) == 0:
        return None
    if len(dets[0]) <= 9:
        covar_matrix = covar_e
    else:
        dets_covar = torch.from_numpy(dets[:,9:])
        dets_covar = torch.reshape(dets_covar, (-1, 3))
        u_matrix = torch.zeros((dets_covar.shape[0], 2, 2))
        u_matrix[:, 0, 0] = torch.exp(dets_covar[:, 0])
        u_matrix[:, 0, 1] = dets_covar[:, 1]
        u_matrix[:, 1, 1] = torch.exp(dets_covar[:, 2])
        sigma_inverse = torch.matmul(torch.transpose(u_matrix, 1, 2), u_matrix)
        covar_matrix = torch.linalg.inv(sigma_inverse)
        if covar_e != None and covar_a != None:
            covar_matrix = covar_e + (0.5 * covar_a + 0.5 * covar_matrix) * 100.0
        else:
            covar_matrix = covar_matrix * 100.0
    predicted_multivariate_normal_dists = torch.distributions.multivariate_normal.MultivariateNormal(dets_loc, covariance_matrix = covar_matrix)
    negative_log_prob = - \
        predicted_multivariate_normal_dists.log_prob(gt_loc)
    #negative_log_prob_mean = negative_log_prob.mean()
    return negative_log_prob.tolist()

def compute_reg_calibrated(dets, gt, covar_e = None, covar_a = None, w=0.0):
    dets_loc = torch.from_numpy(dets[:,:8])
    dets_loc = torch.reshape(dets_loc, (-1, 2))
    gt_loc = torch.from_numpy(gt)
    gt_loc = torch.reshape(gt_loc, (-1, 2))
    if len(dets) == 0:
        return None
    if len(dets[0]) <= 9:
        covar_matrix = covar_e
        calibrated = []
        for i in range(len(dets_loc)):
            predicted_multivariate_normal_dists = multivariate_normal(mean = dets_loc[i], cov = covar_matrix)
            calibrated.append(predicted_multivariate_normal_dists.cdf(gt_loc[i]))
        return calibrated
    else:
        dets_covar = torch.from_numpy(dets[:,9:])
        dets_covar = torch.reshape(dets_covar, (-1, 3))
        u_matrix = torch.zeros((dets_covar.shape[0], 2, 2))
        u_matrix[:, 0, 0] = torch.exp(dets_covar[:, 0])
        u_matrix[:, 0, 1] = dets_covar[:, 1]
        u_matrix[:, 1, 1] = torch.exp(dets_covar[:, 2])
        sigma_inverse = torch.matmul(torch.transpose(u_matrix, 1, 2), u_matrix)
        covar_matrix = torch.linalg.inv(sigma_inverse)
        if covar_e != None and covar_a != None:
            covar_matrix = covar_e + (0.5 * covar_a + 0.5 * covar_matrix) * 100.0
        else:
            covar_matrix = covar_matrix * 100.0
    calibrated = []
    for i in range(len(dets_loc)):
        predicted_multivariate_normal_dists = multivariate_normal(mean = dets_loc[i], cov = covar_matrix[i])
        calibrated.append(predicted_multivariate_normal_dists.cdf(gt_loc[i]))
    return calibrated

def average_precision(recalls, precisions, mode="area"):
    """Calculate average precision (for single or multiple scales).
    Args:
        recalls (ndarray): shape (num_scales, num_dets) or (num_dets, )
        precisions (ndarray): shape (num_scales, num_dets) or (num_dets, )
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]
    Returns:
        float or ndarray: calculated average precision
    """
    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == "area":
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum((mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == "11points":
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
            ap /= 11
    else:
        raise ValueError('Unrecognized mode, only "area" and "11points" are supported')
    if no_scale:
        ap = ap[0]
    return ap


def tpfp_default(
    det_bboxes, gt_bboxes, gt_bboxes_ignore=None, iou_thr=0.5, area_ranges=None
):
    """Check if detected bboxes are true positive or false positive.
    Args:
        det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 4). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. Default: None.
    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
            each array is (num_scales, m).
    """
    # an indicator of ignored gts
    gt_ignore_inds = np.concatenate(
        (
            np.zeros(gt_bboxes.shape[0], dtype=np.bool),
            np.ones(gt_bboxes_ignore.shape[0], dtype=np.bool),
        )
    )
    # stack gt_bboxes and gt_bboxes_ignore for convenience
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))

    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            det_areas = (det_bboxes[:, 2] - det_bboxes[:, 0]) * (
                det_bboxes[:, 3] - det_bboxes[:, 1]
            )
            for i, (min_area, max_area) in enumerate(area_ranges):
                fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
        return tp, fp

    gt_corners = np.zeros((gt_bboxes.shape[0], 4, 2), dtype=np.float32)
    pred_corners = np.zeros((det_bboxes.shape[0], 4, 2), dtype=np.float32)

    for k in range(gt_bboxes.shape[0]):
        gt_corners[k, 0, 0] = gt_bboxes[k][0]
        gt_corners[k, 0, 1] = gt_bboxes[k][1]
        gt_corners[k, 1, 0] = gt_bboxes[k][2]
        gt_corners[k, 1, 1] = gt_bboxes[k][3]
        gt_corners[k, 2, 0] = gt_bboxes[k][4]
        gt_corners[k, 2, 1] = gt_bboxes[k][5]
        gt_corners[k, 3, 0] = gt_bboxes[k][6]
        gt_corners[k, 3, 1] = gt_bboxes[k][7]

    if det_bboxes.ndim == 1:
        det_bboxes = np.array([det_bboxes])

    for k in range(det_bboxes.shape[0]):
        pred_corners[k, 0, 0] = det_bboxes[k][0]
        pred_corners[k, 0, 1] = det_bboxes[k][1]
        pred_corners[k, 1, 0] = det_bboxes[k][2]
        pred_corners[k, 1, 1] = det_bboxes[k][3]
        pred_corners[k, 2, 0] = det_bboxes[k][4]
        pred_corners[k, 2, 1] = det_bboxes[k][5]
        pred_corners[k, 3, 0] = det_bboxes[k][6]
        pred_corners[k, 3, 1] = det_bboxes[k][7]

    gt_box = convert_format(gt_corners)
    pred_box = convert_format(pred_corners)
    save_flag = False
    for gt in gt_box:
        iou = np.array(compute_iou(gt, pred_box))
        if not save_flag:
            box_iou = iou
            save_flag = True
        else:
            box_iou = np.vstack((box_iou, iou))

    # make dimension the same
    if len(gt_box) == 1:
        box_iou = np.array([box_iou])

    ious = box_iou.T
    #    ious = bbox_overlaps(det_bboxes, gt_bboxes)
    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)

    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, 8])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1]
            )
            gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore_inds[matched_gt] or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                    else:
                        fp[k, i] = 1
                # otherwise ignore this detected bbox, tp = 0, fp = 0
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :4]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    return tp, fp


def eval_map(
    det_results,
    annotations,
    scale_ranges=None,
    iou_thr=0.5,
    dataset=None,
    logger=None,
    nproc=4,
):
    """Evaluate mAP of a dataset.
    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:
            - `bboxes`: numpy array of shape (n, 4)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 4)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple] | None): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Default: None.
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        dataset (list[str] | str | None): Dataset name or dataset classes,
            there are minor differences in metrics for different datsets, e.g.
            "voc07", "imagenet_det", etc. Default: None.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.
    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)
    #assert (len(det_results[0][0][0]) == 21) or (len(det_results[0][0][0]) == 9)

    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = (
        [(rg[0] ** 2, rg[1] ** 2) for rg in scale_ranges]
        if scale_ranges is not None
        else None
    )

    pool = Pool(nproc)
    eval_results = []
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results(det_results, annotations, i)
        tpfp_func = tpfp_default
        # compute tp and fp for each image with multiple processes
        tpfp = pool.starmap(
            tpfp_func,
            zip(
                cls_dets,
                cls_gts,
                cls_gts_ignore,
                [iou_thr for _ in range(num_imgs)],
                [area_ranges for _ in range(num_imgs)],
            ),
        )
        tp, fp = tuple(zip(*tpfp))
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for j, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                gt_areas = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area) & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, 8])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = "area" if dataset != "voc07" else "11points"
        ap = average_precision(recalls, precisions, mode)
        eval_results.append(
            {
                "num_gts": num_gts,
                "num_dets": num_dets,
                "recall": recalls,
                "precision": precisions,
                "ap": ap,
            }
        )
    pool.close()
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result["ap"] for cls_result in eval_results])
        all_num_gts = np.vstack([cls_result["num_gts"] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result["num_gts"] > 0:
                aps.append(cls_result["ap"])
        mean_ap = np.array(aps).mean().item() if aps else 0.0

    print_map_summary(mean_ap, eval_results, dataset, area_ranges, logger=logger)

    return mean_ap, eval_results


def get_cls_results(det_results, annotations, class_id):
    """Get det results and gt information of a certain class.
    Args:
        det_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.
    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
    """
    cls_dets = [img_res[class_id] for img_res in det_results]
    cls_gts = []
    cls_gts_ignore = []
    for ann in annotations:
        gt_inds = ann["labels"] == class_id

        if ann["bboxes"].ndim == 1:
            cls_gts.append(ann["bboxes"][0])
        else:
            cls_gts.append(ann["bboxes"][gt_inds, :])

        if ann.get("labels_ignore", None) is not None:
            ignore_inds = ann["labels_ignore"] == class_id

            if ann["bboxes_ignore"].ndim == 1:
                cls_gts_ignore.append(ann["bboxes_ignore"][0])
            else:
                cls_gts_ignore.append(ann["bboxes_ignore"][ignore_inds, :])
        else:
            cls_gts_ignore.append(np.empty((0, 8), dtype=np.float32))

    return cls_dets, cls_gts, cls_gts_ignore


def print_map_summary(mean_ap, results, dataset=None, scale_ranges=None, logger=None):
    """Print mAP and results of each class.
    A table will be printed to show the gts/dets/recall/AP of each class and
    the mAP.
    Args:
        mean_ap (float): Calculated from `eval_map()`.
        results (list[dict]): Calculated from `eval_map()`.
        dataset (list[str] | str | None): Dataset name or dataset classes.
        scale_ranges (list[tuple] | None): Range of scales to be evaluated.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.
    """

    if logger == "silent":
        return

    if isinstance(results[0]["ap"], np.ndarray):
        num_scales = len(results[0]["ap"])
    else:
        num_scales = 1

    if scale_ranges is not None:
        assert len(scale_ranges) == num_scales

    num_classes = len(results)

    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    aps = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    for i, cls_result in enumerate(results):
        if cls_result["recall"].size > 0:
            recalls[:, i] = np.array(cls_result["recall"], ndmin=2)[:, -1]
        aps[:, i] = cls_result["ap"]
        num_gts[:, i] = cls_result["num_gts"]

    if dataset is None:
        label_names = [str(i) for i in range(num_classes)]
    else:
        label_names = dataset

    if not isinstance(mean_ap, list):
        mean_ap = [mean_ap]

    header = ["class", "gts", "dets", "recall", "ap"]
    for i in range(num_scales):
        if scale_ranges is not None:
            print_log(f"Scale range {scale_ranges[i]}", logger=logger)
        table_data = [header]
        for j in range(num_classes):
            row_data = [
                label_names[j],
                num_gts[i, j],
                results[j]["num_dets"],
                f"{recalls[i, j]:.3f}",
                f"{aps[i, j]:.3f}",
            ]
            table_data.append(row_data)
        table_data.append(["mAP", "", "", "", f"{mean_ap[i]:.3f}"])
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log("\n" + table.table, logger=logger)

def match_tp_fp(
    det_bboxes, gt_bboxes, gt_bboxes_ignore=None, iou_thr=0.5, area_ranges=None
):
    """Check if detected bboxes are true positive or false positive.
    Args:
        det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 4). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. Default: None.
    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
            each array is (num_scales, m).
    """
    # an indicator of ignored gts
    gt_ignore_inds = np.concatenate(
        (
            np.zeros(gt_bboxes.shape[0], dtype=np.bool),
            np.ones(gt_bboxes_ignore.shape[0], dtype=np.bool),
        )
    )
    # stack gt_bboxes and gt_bboxes_ignore for convenience
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))

    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
        
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.int)
    fp = np.zeros((num_scales, num_dets), dtype=np.int)

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            det_areas = (det_bboxes[:, 2] - det_bboxes[:, 0]) * (
                det_bboxes[:, 3] - det_bboxes[:, 1]
            )
            for i, (min_area, max_area) in enumerate(area_ranges):
                fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
        return tp, fp

    gt_corners = np.zeros((gt_bboxes.shape[0], 4, 2), dtype=np.float32)
    pred_corners = np.zeros((det_bboxes.shape[0], 4, 2), dtype=np.float32)

    for k in range(gt_bboxes.shape[0]):
        gt_corners[k, 0, 0] = gt_bboxes[k][0]
        gt_corners[k, 0, 1] = gt_bboxes[k][1]
        gt_corners[k, 1, 0] = gt_bboxes[k][2]
        gt_corners[k, 1, 1] = gt_bboxes[k][3]
        gt_corners[k, 2, 0] = gt_bboxes[k][4]
        gt_corners[k, 2, 1] = gt_bboxes[k][5]
        gt_corners[k, 3, 0] = gt_bboxes[k][6]
        gt_corners[k, 3, 1] = gt_bboxes[k][7]

    if det_bboxes.ndim == 1:
        det_bboxes = np.array([det_bboxes])

    for k in range(det_bboxes.shape[0]):
        pred_corners[k, 0, 0] = det_bboxes[k][0]
        pred_corners[k, 0, 1] = det_bboxes[k][1]
        pred_corners[k, 1, 0] = det_bboxes[k][2]
        pred_corners[k, 1, 1] = det_bboxes[k][3]
        pred_corners[k, 2, 0] = det_bboxes[k][4]
        pred_corners[k, 2, 1] = det_bboxes[k][5]
        pred_corners[k, 3, 0] = det_bboxes[k][6]
        pred_corners[k, 3, 1] = det_bboxes[k][7]

    gt_box = convert_format(gt_corners)
    pred_box = convert_format(pred_corners)
    save_flag = False
    for gt in gt_box:
        iou = np.array(compute_iou(gt, pred_box))
        if not save_flag:
            box_iou = iou
            save_flag = True
        else:
            box_iou = np.vstack((box_iou, iou))

    # make dimension the same
    if len(gt_box) == 1:
        box_iou = np.array([box_iou])

    ious = box_iou.T
    #    ious = bbox_overlaps(det_bboxes, gt_bboxes)
    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)

    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, 8])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1]
            )
            gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore_inds[matched_gt] or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = matched_gt
                    else:
                        fp[k, i] = 1
                # otherwise ignore this detected bbox, tp = 0, fp = 0
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :4]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    return tp, fp

def compute_residual_error(dets, gt):
    assert len(dets) == len(gt)
    dets_loc = torch.from_numpy(dets[:,:8])
    dets_loc = torch.reshape(dets_loc, (-1, 2))
    gt_loc = torch.from_numpy(gt)
    gt_loc = torch.reshape(gt_loc, (-1, 2))
    return (gt_loc - dets_loc).tolist()

def get_predicted_covar(dets):
    dets_covar = dets[:,9:]
    dets_covar = np.reshape(dets_covar, (-1, 3))
    u_matrix = np.zeros((dets_covar.shape[0], 2, 2))
    u_matrix[:, 0, 0] = np.exp(dets_covar[:, 0])
    u_matrix[:, 0, 1] = dets_covar[:, 1]
    u_matrix[:, 1, 1] = np.exp(dets_covar[:, 2])
    sigma_inverse = np.matmul(u_matrix.transpose((0,2,1)), u_matrix)
    covar_matrix = np.linalg.inv(sigma_inverse)
    return covar_matrix.tolist()

def compute_reg_entropy(dets, covar_e = None):
    dets_loc = torch.from_numpy(dets[:,:8])
    dets_loc = torch.reshape(dets_loc, (-1, 2))
    if len(dets[0]) <= 9:
        covar_matrix = covar_e
    else:
        dets_covar = torch.from_numpy(dets[:,9:])
        dets_covar = torch.reshape(dets_covar, (-1, 3))
        u_matrix = torch.zeros((dets_covar.shape[0], 2, 2))
        u_matrix[:, 0, 0] = torch.exp(dets_covar[:, 0])
        u_matrix[:, 0, 1] = dets_covar[:, 1]
        u_matrix[:, 1, 1] = torch.exp(dets_covar[:, 2])
        sigma_inverse = torch.matmul(torch.transpose(u_matrix, 1, 2), u_matrix)
        covar_matrix = torch.linalg.inv(sigma_inverse) + 1e-2 * torch.eye(2)
    predicted_multivariate_normal_dists = torch.distributions.multivariate_normal.MultivariateNormal(dets_loc, covariance_matrix = covar_matrix)
    total_entropy = predicted_multivariate_normal_dists.entropy()
    #total_entropy_mean = total_entropy.mean()
    return total_entropy.tolist()

def eval_nll(
    det_results,
    annotations,
    scale_ranges=None,
    iou_thr=0.5,
    logger=None,
    nproc=4,
    covar_e = None,
    covar_a = None,
    w=0.0
):
    """Evaluate mAP of a dataset.
    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:
            - `bboxes`: numpy array of shape (n, 4)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 4)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple] | None): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Default: None.
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        dataset (list[str] | str | None): Dataset name or dataset classes,
            there are minor differences in metrics for different datsets, e.g.
            "voc07", "imagenet_det", etc. Default: None.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.
    Returns:
        tuple: [[tp_nll, fp_entropy]]
    """
    assert len(det_results) == len(annotations)
    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = (
        [(rg[0] ** 2, rg[1] ** 2) for rg in scale_ranges]
        if scale_ranges is not None
        else None
    )

    pool = Pool(nproc)
    eval_results = []
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results(det_results, annotations, i)
        tp_nll = []
        fp_entropy = []
        tpfp_func = match_tp_fp
        # compute tp and fp for each image with multiple processes
        tpfp = pool.starmap(
            tpfp_func,
            zip(
                cls_dets,
                cls_gts,
                cls_gts_ignore,
                [iou_thr for _ in range(num_imgs)],
                [area_ranges for _ in range(num_imgs)],
            ),
        )
        tp_all, fp_all = list(zip(*tpfp))
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for j, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                gt_areas = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area) & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        for dets, gt, match, fp in zip(cls_dets, cls_gts, tp_all, fp_all):
            tp = np.squeeze((1 - fp).astype(bool))
            fp = np.squeeze(fp.astype(bool))
            match = np.squeeze(match)
            if tp.shape is ():
                tp = np.array([tp])
                match = np.array([match])
            if fp.shape is ():
                fp = np.array([fp])
            if len(tp) != 0:
                tp_dets = dets[tp]
                tp_match = match[tp]
                tp_gt = gt[tp_match]
                nll = compute_reg_nll(tp_dets, tp_gt, covar_e, covar_a, w)
                if nll != None:
                    tp_nll.extend(nll)
            #if len(dets[fp]) != 0:
            #    fp_dets = dets[fp]
            #    entropy = compute_reg_entropy(fp_dets)
            #    fp_entropy.extend(entropy)
        eval_results.append({
            "num_gts": num_gts,
            "NLL": np.mean(tp_nll),
            #"FP_entropy": np.mean(fp_entropy)
        })
    return eval_results

def match_pairs(
    det_results,
    annotations,
    scale_ranges=None,
    iou_thr=0.5
):
    """Evaluate mAP of a dataset.
    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:
            - `bboxes`: numpy array of shape (n, 4)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 4)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple] | None): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Default: None.
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
    Returns:
        list: tp_results [[predicted, ground truth]]
        list: fp_results [predicted]
    """
    assert len(det_results) == len(annotations)
    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = (
        [(rg[0] ** 2, rg[1] ** 2) for rg in scale_ranges]
        if scale_ranges is not None
        else None
    )

    pool = Pool(4)
    tp_results = []
    fp_results = []
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results(det_results, annotations, i)
        tpfp_func = match_tp_fp
        # compute tp and fp for each image with multiple processes
        tpfp = pool.starmap(
            tpfp_func,
            zip(
                cls_dets,
                cls_gts,
                cls_gts_ignore,
                [iou_thr for _ in range(num_imgs)],
                [area_ranges for _ in range(num_imgs)],
            ),
        )
        tp_all, fp_all = list(zip(*tpfp))
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for j, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                gt_areas = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area) & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        for dets, gt, match, fp in zip(cls_dets, cls_gts, tp_all, fp_all):
            tp = np.squeeze((1 - fp).astype(bool))
            fp = np.squeeze(fp.astype(bool))
            match = np.squeeze(match)
            if tp.shape is ():
                tp = np.array([tp])
                match = np.array([match])
            if fp.shape is ():
                fp = np.array([fp])
            if len(tp) != 0:
                tp_dets = dets[tp]
                tp_match = match[tp]
                tp_gt = gt[tp_match]
                assert len(tp_dets) == len(tp_gt)
                tp_results.append([tp_dets, tp_gt])
            if len(dets[fp]) != 0: 
                fp_dets = dets[fp]
                fp_results.append(fp_dets)   
    return tp_results, fp_results

def eval_calibrate(
    det_results,
    annotations,
    scale_ranges=None,
    iou_thr=0.5,
    logger=None,
    nproc=4,
    covar_e = None,
    covar_a = None,
    w=0.0
):
    assert len(det_results) == len(annotations)
    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = (
        [(rg[0] ** 2, rg[1] ** 2) for rg in scale_ranges]
        if scale_ranges is not None
        else None
    )

    pool = Pool(nproc)
    eval_results = []
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results(det_results, annotations, i)
        tp_cal = []
        tpfp_func = match_tp_fp
        # compute tp and fp for each image with multiple processes
        tpfp = pool.starmap(
            tpfp_func,
            zip(
                cls_dets,
                cls_gts,
                cls_gts_ignore,
                [iou_thr for _ in range(num_imgs)],
                [area_ranges for _ in range(num_imgs)],
            ),
        )
        tp_all, fp_all = list(zip(*tpfp))
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for j, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                gt_areas = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area) & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        for dets, gt, match, fp in zip(cls_dets, cls_gts, tp_all, fp_all):
            tp = np.squeeze((1 - fp).astype(bool))
            fp = np.squeeze(fp.astype(bool))
            match = np.squeeze(match)
            if tp.shape is ():
                tp = np.array([tp])
                match = np.array([match])
            if fp.shape is ():
                fp = np.array([fp])
            if len(tp) != 0:
                tp_dets = dets[tp]
                tp_match = match[tp]
                tp_gt = gt[tp_match]
                cal = compute_reg_calibrated(tp_dets, tp_gt, covar_e, covar_a, w)
                if cal != None:
                    tp_cal.extend(cal)
            #if len(dets[fp]) != 0:
            #    fp_dets = dets[fp]
            #    entropy = compute_reg_entropy(fp_dets)
            #    fp_entropy.extend(entropy)
        histogram_bin_step_size = 1 / 15.0
        reg_calibration_error = []
        expected = np.arange( histogram_bin_step_size, 1.0 + histogram_bin_step_size, histogram_bin_step_size)
        predicted = []
        for i in expected:
            elements_in_bin = (tp_cal <= i)
            num_elems_in_bin_i = sum(elements_in_bin) * 1.0
            predicted.append(num_elems_in_bin_i / len(tp_cal))
            reg_calibration_error.append( (num_elems_in_bin_i / len(tp_cal) - i ) ** 2)
        eval_results.append({
            "num_gts": num_gts,
            "ECE": np.mean(reg_calibration_error),
            "exp_cal": expected,
            "pre_cal": predicted
        })
    return eval_results


def get_residual_error_and_cov(
    det_results,
    annotations,
    scale_ranges=None,
    iou_thr=0.0,
    logger=None,
    nproc=4,
):
    """Get residual error between the predicted bounding box and the ground-truth bounding box.
    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:
            - `bboxes`: numpy array of shape (n, 4)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 4)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple] | None): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Default: None.
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        dataset (list[str] | str | None): Dataset name or dataset classes,
            there are minor differences in metrics for different datsets, e.g.
            "voc07", "imagenet_det", etc. Default: None.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.
    Returns:
        resError: y-\hat{y}
    """
    assert len(det_results) == len(annotations)
    covar_flag = len(det_results[0][0][0]) == 21
    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = (
        [(rg[0] ** 2, rg[1] ** 2) for rg in scale_ranges]
        if scale_ranges is not None
        else None
    )

    pool = Pool(nproc)
    res_error = []
    predicted_covar = []
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results(det_results, annotations, i)
        one_res_error = []
        one_covar = []
        tpfp_func = match_tp_fp
        # compute tp and fp for each image with multiple processes
        tpfp = pool.starmap(
            tpfp_func,
            zip(
                cls_dets,
                cls_gts,
                cls_gts_ignore,
                [iou_thr for _ in range(num_imgs)],
                [area_ranges for _ in range(num_imgs)],
            ),
        )
        tp_all, fp_all = list(zip(*tpfp))
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for j, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                gt_areas = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area) & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        for dets, gt, match, fp in zip(cls_dets, cls_gts, tp_all, fp_all):
            tp = np.squeeze((1 - fp).astype(bool))
            match = np.squeeze(match)
            if tp.shape is ():
                tp = np.array([tp])
                match = np.array([match])
            if len(tp) != 0:
                tp_dets = dets[tp]
                tp_match = match[tp]
                tp_gt = gt[tp_match]
                one_res_error.extend(compute_residual_error(tp_dets, tp_gt))
                if covar_flag:
                    one_covar.extend(get_predicted_covar(tp_dets))
        res_error.append(one_res_error)
        predicted_covar.append(one_covar)
    if covar_flag:
        return res_error, predicted_covar
    else:
        return res_error, None