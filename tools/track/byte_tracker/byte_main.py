"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import shutil
import numpy as np
import math
import ipdb
from byte_tracker import BYTETracker

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from skimage import io

import time
import argparse

np.random.seed(0)

def order_det_res(root):
    files = os.listdir(root)
    for file in files:
        F = os.path.join(root, file)
        if F[-4:] != ".txt":
            continue
        with open(F, "r") as f:
            lines = f.readlines()
        lines.sort(key=lambda x: int(x.split(",")[0]))
        with open(F, "w") as F:
            for line in lines:
                F.write(line)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="SORT demo")
    parser.add_argument("--mode")  # TODO: what is mode
    parser.add_argument("--track_thresh", type=float, default=0.6, help="detection confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument(
        "--nll_threshold", help="Maximum for match.", type=float, default=10
    )
    # parser.add_argument('--save_path', type=str)
    parser.add_argument(
        "--display",
        dest="display",
        help="Display online tracker output (slow) [False]",
        action="store_true",
    )
    # parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    # parser.add_argument(
    #     "--max_age",
    #     help="Maximum number of frames to keep alive a track without associated detections.",
    #     type=int,
    #     default=1,
    # )
    # parser.add_argument(
    #     "--min_hits",
    #     help="Minimum number of associated detections before track is initialised.",
    #     type=int,
    #     default=3,
    # )
    # parser.add_argument(
    #     "--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3
    # )
    parser.add_argument("--scene_idxes_file", type=str, help="File containing idxes of scenes to run tracking")
    parser.add_argument(
        "--from_agent", default=0, type=int, help="start from which agent"
    )
    parser.add_argument(
        "--to_agent", default=6, type=int, help="until which agent (index + 1)"
    )
    parser.add_argument(
        "--det_logs_path", default='', type=str, help="Det logs path (to get the tracking input)"
    )
    parser.add_argument("--split", type=str, help="[test/val]")
    parser.add_argument("--output_cov", default=False, action="store_true", help = "Enable to use variance of x,y as input of Filter for kalman filter")
    parser.add_argument("--nll_ass", default=False, action="store_true", help = "Enable to use variance of x,y as input of Filter for association")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    display = args.display
    scene_idxes_file = open(args.scene_idxes_file, "r")
    scene_idxes = [int(line.strip()) for line in scene_idxes_file]
    print(f'scenes to run: {scene_idxes}')
    #ipdb.set_trace()

    for current_agent in range(args.from_agent, args.to_agent):
        total_time = 0.0
        total_frames = 0
        root = os.path.join(os.path.join(args.det_logs_path, f'{args.mode}/tracking{current_agent}'))
        order_det_res(root)
        det_results = os.listdir(root)
        save_path = f"./{args.mode}/agent{current_agent}"
        os.makedirs(save_path, exist_ok=True)
        for seq in det_results:
            if seq[-4:] != ".txt":
                continue
            mot_tracker = BYTETracker(
                args
            )  # create instance of the SORT tracker
            seq_dets = np.loadtxt(os.path.join(root, seq), delimiter=",")
            with open(os.path.join(save_path, seq.replace("det_", "")), "w") as out_file:
                if len(seq_dets) == 0:
                    continue

                print("Processing %s." % (os.path.join(root, seq)))
                for frame in range(int(seq_dets[:, 0].max())):
                    frame += 1  # detection and frame numbers begin at 1
                    if args.output_cov:
                        dets = seq_dets[seq_dets[:, 0] == frame, 2:]
                    else:
                        dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
                    dets[:, 2:4] += dets[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
                    total_frames += 1

                    start_time = time.time()
                    trackers = mot_tracker.update(dets)
                    cycle_time = time.time() - start_time
                    total_time += cycle_time

                    for track in trackers:
                        tlwh = track.tlwh
                        track_id = track.track_id
                        print(
                            "%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1"
                            % (frame, track_id, tlwh[0], tlwh[1], tlwh[2], tlwh[3]),
                            file=out_file,
                        )

        eval_dir = f"../TrackEval/data/trackers/mot_challenge/V2X-{args.split}{current_agent}/sort-{args.mode}/data"
        os.makedirs(eval_dir, exist_ok=True)
        for seq in scene_idxes:
            tracker_txt = os.path.join(save_path, f"{seq}.txt")
            if not os.path.exists(tracker_txt):
                continue

            shutil.copy(tracker_txt, os.path.join(eval_dir, f"{seq}.txt"))
        print(
            "Total Tracking took: %.3f seconds for %d frames or %.1f FPS"
            % (total_time, total_frames, total_frames / total_time)
        )
        print("Saved results to ", eval_dir)
