import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from kalman_filter import KalmanFilter
import matching
from basetrack import BaseTrack, TrackState
from kalman_filter_uq import KalmanFilterUQ

var_cp_dict = {"upperbound": [156.3751341925818, 65.24260517866209, 2785.7870238883734, 26.43169171689354], "disco": [214.13191750615232, 77.15047953920649, 2503.1216487206066, 24.387592952924123], "lowerbound": [268.0828005828979, 93.20827260609053, 18277.14814591181, 40.319136673153885]}

std_cp_dict = {"upperbound":[12.505004365956127, 8.077289469782675, 52.780555357900255, 5.14117610249771], "disco": [14.633246991223524, 8.783534569819059, 50.031206748594485, 4.938379587772099], "lowerbound": [16.373234273743776, 9.654443153599825, 135.19300331715326, 6.34973516559186]}

output_cov = False
nll_ass = False

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    shared_kalman_uq = KalmanFilterUQ()
    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh[:4], dtype=np.float)
        if output_cov or nll_ass:
            self._stdrh = np.asarray(tlwh[4:], dtype = np.float)
        else:
            self._stdrh = None
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance, self._stdrh)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            if output_cov:
                multi_std = np.asarray([st._stdrh for st in stracks])
                multi_mean, multi_covariance = STrack.shared_kalman_uq.multi_predict(multi_mean, multi_covariance, multi_std) 
            else:
                multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh), self._stdrh)

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh), new_track._stdrh
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh), new_track._stdrh)
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    # @jit(nopython=True)
    def stdrh(self):
        return self._stdrh

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:4] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:4] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:4] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilterUQ() if args.output_cov else KalmanFilter()
        self.mode = args.mode.split("/")[0]
        global output_cov, nll_ass
        output_cov = args.output_cov
        nll_ass = args.nll_ass
        self.nll_threshold = args.nll_threshold
        self.matched_num = 0
        self.unmatched_det = 0
        self.unmatched_trk = 0
    
    def return_match_result(self):
        return self.matched_num, self.unmatched_det, self.unmatched_trk

    def compute_variance(self, log_var):
        out = np.exp(log_var) * np.array(var_cp_dict[self.mode])
        return out
    
    def compute_std(self, log_var):
        out = np.sqrt(np.exp(log_var)) * np.array(std_cp_dict[self.mode])
        return out

    def update(self, output_results):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        scores = output_results[:, 4]
        bboxes = output_results[:, :4]
        if output_cov or nll_ass:
            bboxes = np.concatenate((bboxes, self.compute_std(output_results[:, 8:12])), axis=1)

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        '''Step 4: association with NLL'''
        if nll_ass:
            r_tracked_stracks_temp = [r_tracked_stracks[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
            ind_third = scores <= 0.1
            dets_third = bboxes[ind_third]
            scores_third = scores[ind_third]
            detections_third = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                            (tlbr, s) in zip(dets_third, scores_third)]
            start_det = len(detections_third)
            for i in u_detection:
                detections_third.append(detections[i])
            end_det = len(detections_third)
            for i in u_detection_second:
                detections_third.append(detections_second[i])
            if len(r_tracked_stracks_temp) != 0 and len(detections_third) != 0:
                r_tracked_stracks = r_tracked_stracks_temp
                nll_distance = matching.nll_distance(r_tracked_stracks, detections_third)
                matches, u_track, u_detection_third = matching.linear_assignment(nll_distance, thresh=np.inf)
                removed_u_track = []
                for itracked, idet in matches:
                    if nll_distance[itracked, idet] > self.nll_threshold:
                        u_track = np.append(u_track, itracked)
                        u_detection_third = np.append(u_detection_third, idet)
                        continue
                    if start_det <= idet < end_det:
                        removed_u_track.append(idet-start_det)
                    track = r_tracked_stracks[itracked]
                    det = detections_third[idet]
                    if track.state == TrackState.Tracked:
                        track.update(det, self.frame_id)
                        activated_starcks.append(track)
                    else:
                        track.re_activate(det, self.frame_id, new_id = False)
                        refind_stracks.append(track)
                u_detection_temp = []
                for i in range(len(u_detection)):
                    if i not in removed_u_track:
                        u_detection_temp.append(u_detection[i])
                u_detection = u_detection_temp
                
        
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
        
        self.matched_num += len(activated_starcks) + len(refind_stracks)
        self.unmatched_det += (output_results.shape[0] - (len(activated_starcks) + len(refind_stracks)))
        
        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
        
        self.unmatched_trk += len(lost_stracks) + len(removed_stracks)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        # for track in output_stracks:
        #     tlwh = track.tlwh
        #     track_id = track.track_id
        #     print(tlwh)

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
