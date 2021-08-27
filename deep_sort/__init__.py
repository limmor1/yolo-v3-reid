from .deep_sort import DeepSort


__all__ = ['DeepSort', 'build_tracker']


def build_tracker(cfg, use_cuda):
    return DeepSort(cfg.DEEPSORT.REID_CKPT,
                max_dist=cfg.DEEPSORT.MAX_DIST,  # max cosine distance = 0.2
                min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,  # min detector confidence = 0.3
                nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,  # suppress all overlapping bboxes above this IOU = 0.5
                max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,  # 0.7  TODO
                max_age=cfg.DEEPSORT.MAX_AGE,  #   max number of missed misses before a track is deleted = 70
                n_init=cfg.DEEPSORT.N_INIT,  # number of consecutive detections before a track is confirmed (else - "deleted") = 3
                nn_budget=cfg.DEEPSORT.NN_BUDGET,  # max number of samples (per target_id [TODO: n.b. not class!]) to compare with, circular buffer = 100
                use_cuda=use_cuda)
    









