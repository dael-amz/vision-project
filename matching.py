import numpy as np
import cv2
import skimage.feature as features
import math

class Keypoints():
    locations: np.array
    descriptors: np.array
    scales: np.array

    def __init__(self, locations, descriptors, scales):
        self.locations = locations
        self.descriptors = descriptors
        self.scales = scales

    def loc(self):
        return self.locations
    
    def desc(self):
        return self.descriptors
    
    def sc(self):
        return self.scales
    
    def keypoint(self, index):
        kp = {}
        kp['location'] = self.locations[index]
        kp['descriptors'] = self.descriptors[index]
        kp['scale'] = self.scales[index]

        return kp

class D_MATCHER():
    distorted_kps: Keypoints
    origin_kps: Keypoints
    scale_thresh: float
    pos_thresh: float
    distortion_func: callable
    matches: np.array
    s_true: np.array
    M_true: np.array

    def __init__(self,
                 distorted_kps: None,
                 origin_kps: None,
                 scale_thresh: 0.5,
                 pos_thresh: 3.0,
                 distortion_func = lambda x,y: (x,y)):
        
        self.distorted_kps = distorted_kps
        self.origin_kps = origin_kps
        self.scale_thresh = scale_thresh
        self.pos_thresh = pos_thresh
        self.distortion_func = distortion_func

    def match_kp(self, max_ratio = 0.8, cross_check = True):
        origin_desc = self.origin_kps.desc()
        distorted_desc = self.distorted_kps.desc()
        self.matches = features.match_descriptors(origin_desc,
                                            distorted_desc,
                                            cross_check=cross_check,
                                            max_ratio=max_ratio)
        return self.matches
    
    def compare_scale(self, distorted, origin):
        return (np.abs(np.log2(distorted) - np.log2(origin))) <= self.scale_thresh
    
    def unique_assignment(
        self,
        query_idx: np.array,
        train_idx: np.array,
        score: np.array,):
        """
        Enforce one-to-one correspondences greedily by lowest score first.
        Mostly useful for detector repeatability pairing.
        """
        order = np.argsort(score)
        used_q = set()
        used_t = set()
        keep = []

        for k in order:
            q = int(query_idx[k])
            t = int(train_idx[k])
            if q in used_q or t in used_t:
                continue
            used_q.add(q)
            used_t.add(t)
            keep.append(k)

        keep = np.asarray(keep, dtype=int)
        return query_idx[keep], train_idx[keep], score[keep]

    def compute_s_true(self, one_to_one = True):
        
        distorted_loc = self.distorted_kps.loc()
        origin_loc = self.origin_kps.loc()

        distorted_scales = self.distorted_kps.sc()[np.newaxis, :]
        origin_scales = self.origin_kps.sc()

        dist_frame_origin_loc, dist_frame_origin_scales = self.distortion_func(origin_loc, origin_scales)
        dist_frame_origin_scales = dist_frame_origin_scales[:, np.newaxis]

        diff = dist_frame_origin_loc[:, np.newaxis ,:] - distorted_loc[np.newaxis, :, :]
        norms = np.linalg.norm(diff, axis = 2)

        scales = self.compare_scale(dist_frame_origin_scales, distorted_scales)

        c_matrix = (norms <= self.pos_thresh) & scales
        row, col = np.where(c_matrix)

        q_idx = row.astype(int)
        t_idx = col.astype(int)
        score = norms[row, col]

        if one_to_one:
            q_idx, t_idx, score = self.unique_assignment(q_idx, t_idx, score)

        self.s_true = np.column_stack((q_idx, t_idx))

        return self.s_true

    def true_matches(self):

        distorted_idx = self.matches[:, 1]
        origin_idx = self.matches[:, 0]

        distorted_loc = self.distorted_kps.loc()[distorted_idx]
        origin_loc = self.origin_kps.loc()[origin_idx]

        distorted_scales = self.distorted_kps.sc()[distorted_idx]
        origin_scales = self.origin_kps.sc()[origin_idx]

        dist_frame_origin_loc, dist_frame_origin_scales = self.distortion_func(origin_loc, origin_scales)

        diff = dist_frame_origin_loc- distorted_loc
        norms = np.linalg.norm(diff, axis = 1)

        scales = self.compare_scale(dist_frame_origin_scales, distorted_scales)

        c_matrix = (norms <= self.pos_thresh) #& scales
        self.M_true = self.matches[c_matrix]

        return self.M_true
    
    def compute_stats(self, max_ratio = 0.8, cross_check = True):

        self.match_kp(max_ratio=max_ratio, cross_check=cross_check)

        self.compute_s_true()
        self.true_matches()

        result = {
            'repeatability': len(self.s_true) / min(len(self.distorted_kps.loc()), len(self.origin_kps.loc())),
            'recall': len(self.M_true) / len(self.s_true),
            'precision': len(self.M_true) / len(self.matches)
        }
    
        return result


