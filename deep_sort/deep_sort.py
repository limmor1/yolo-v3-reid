import numpy as np
import torch

from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker

import time
__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap

        self.extractor = Extractor(model_path, use_cuda=use_cuda)

        max_cosine_distance = max_dist
        nn_budget = 100
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)
        self.nms_time = 0
        self.reid_standalone_time = 0

    def update(self, bbox_xywh, confidences, img, reids=None, standalone_reid=True):
        self.height, self.width = img.shape[:2]

        # j: generate detection features (if none are supplied by YOLO)
        if standalone_reid:
            start = time.time()
            features = self._get_features(bbox_xywh, img)  # j: matrix of row-vectors of features (n, 512)
            finish = time.time()
            self.reid_standalone_time += finish - start

        else:
            features = reids  # j: when YOLO-v3-REID is used, reid [tensor[1,128,13,13], tensor[1,128,26,26], tensor[1,128,52,52]]
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)

        # j: YOLO-REID: turned off confidence filtering, since YOLO-REID already has it
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i,conf in enumerate(confidences)]  # if conf>self.min_confidence]
        # j: Detection = tlwh bbox, features, confidence

        start = time.time()
        # j: YOLO-REID: turned off nms, since YOLO-REID already has it
        # run on non-maximum supression
        # j: cleanup detections by confidence and non-max suppression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        finish = time.time()
        self.nms_time += finish - start
        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.float32))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs


    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        """ j: (x,y,width,height) to (top, left, width, height)"""
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2.
        bbox_tlwh[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2.
        return bbox_tlwh


    def _xywh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x,y,w,h = bbox_tlwh
        x1 = max(x,0)
        x2 = min(x+w,self.width-1)
        y1 = max(y,0)
        y2 = min(y+h,self.height-1)
        return x1,y1,x2,y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1,y1,x2,y2 = bbox_xyxy

        t = x1
        l = y1
        w = x2-x1
        h = y2-y1
        return t,l,w,h

    def padded_to_original_size(self, x1, y1, h_old, w_old, w_new, h_new):
        assert h_old < w_old
        aspect = w_old / h_old
        h_new_ = w_new / aspect  # height without margins
        margin = (h_new - h_new_) / 2  # j: what remains from 416 above new height
        x_scale = w_old / w_new
        y_scale = h_old / h_new_
        x_orig = x1 * x_scale
        y_orig = (y1 - margin) * y_scale
        return x_orig, y_orig
    
    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1,y1,x2,y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2,x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features


