import os
import cv2
import time
import argparse
import warnings

# YOLO-v3 imports
from detector_YOLO_v3 import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results

# YOLO-v3-REID imports
from detector_YOLO_v3_REID.YOLOv3_lindernoren.models import *
from detector_YOLO_v3_REID.YOLOv3_lindernoren.utils.transforms import *
import detector_YOLO_v3_REID.YOLOv3_lindernoren.detect as detect

class VideoTracker(object):
    def __init__(self, cfg, args, video_path):
        self.cfg = cfg
        self.args = args
        self.video_path = video_path
        self.logger = get_logger("root")

        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()

        ########### YOLO-v3 ################
        if not args.reid_yolo:
            print("Using vanilla YOLO-v3, Ziqiang Pei, https://github.com/ZQPei/deep_sort_pytorch")
            self.detector = build_detector(cfg, use_cuda=use_cuda)
        ###################################

        ########### YOLO-v3-REID ################
        if args.reid_yolo:
            print("Using modified YOLO-v3 with ReID head, Julius Raskevicius, https://github.com/limmor1/yolo-v3-reid")
            if args.standalone_reid:
                print("ReID features overwritten. Detections YOLO-v3 with ReID head; Reidentifications - standalone module by Ziqiang Pei, https://github.com/ZQPei/deep_sort_pytorch")

            # load model and weights
            model_cfg_path = os.path.dirname(__file__) + "/detector_YOLO_v3_REID/YOLOv3_lindernoren/config/yolov3_80class_reid_allscale_MOTcombined_nosoft_skip_end.cfg"
            # model_cfg_path = os.path.dirname(__file__) + "/detector_YOLO_v3_REID/YOLOv3_lindernoren/config/yolov3_80class_reid_allscale_MOTcombined_nosoft.cfg"
            # model_cfg_path = os.path.dirname(__file__) + "/detector_YOLO_v3_REID/YOLOv3_lindernoren/config/yolov3_80class_reid_allscale_MOTcombined_nosoft_mixed_end.cfg"

            # TODO: convert to cmd parameter
            model_weights_path = os.path.dirname(__file__) + "/detector_YOLO_v3_REID/YOLOv3_lindernoren/checkpoints/yolov3_reid_skip_end.pth"
            # model_weights_path = "os.path.dirname(__file__) + "/detector_YOLO_v3_REID/YOLOv3_lindernoren/checkpoints/yolov3_reid_split_end.pth"
            # model_weights_path = os.path.dirname(__file__) + "/detector_YOLO_v3_REID/YOLOv3_lindernoren/checkpoints/yolov3_reid_mix_end.pth"
            self.detector = load_model(model_cfg_path, model_weights_path, args.use_cuda)
        ##################################################

        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = ["person"]

    def __enter__(self):
        if self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]

        else:
            assert os.path.isfile(self.video_path), "Path error"
            self.vdo.open(self.video_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()

        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)

            # path of saved video and results
            self.save_video_path = os.path.join(self.args.save_path, "results.avi")

            # create video writer
            # j: try one of the other codes if MJPG is not working (n.b. opencv has issues on linux with all codecs)
            # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 20.0, (self.im_width, self.im_height))

            # logging
            self.logger.info("Save results to {}".format(self.args.save_path))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):

        # TODO: convert to cmd parameter
        # scales = [[13, 13], [26, 26], [52, 52]]
        # scales = [[13, 13]]
        # scales = [[26, 26]]
        scales = [[52, 52]]
        for scale in scales:
            print(f"Processing scale {scale}...")
            self.save_results_path = os.path.join(self.args.save_path, f"results_{scale}.txt")
            w_new = 416  # j: height & width of the input to the network
            h_new = 416

            results = []
            idx_frame = 0

            # j: gather average results
            detector_time = 0
            tracker_time = 0
            n_detections = 0
            n_tracks = 0
            while self.vdo.grab():
                # print(f"Frame {idx_frame}...")
                idx_frame += 1
                if idx_frame % self.args.frame_interval:
                    continue

                _, ori_im = self.vdo.retrieve()
                # im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)  # j: jpegs are in BGR pixel color order, we convert to RGB
                im = ori_im.copy()

                # j: resize input to 416x416
                input_img, _ = transforms.Compose([
                    DEFAULT_TRANSFORMS,
                    Resize(w_new)])(
                    (im, np.zeros((1, 6))))  # j: move indexes to accommodate unique ID
                input_img = np.array(transforms.ToPILImage()(input_img))
                # cv2.imwrite(f"/home/julius-think/Thesis/Code/deep_sort_pytorch/result_test.png", input_img)  # j: preview padded input
                # input_img = im

                ########### YOLOv3_lindernoren-v3 ################
                if args.reid_yolo:


                    # do detection
                    # j: YOLO-REID
                    start = time.time()
                    detections, reids = detect.detect_image(self.detector, input_img, args.use_cuda, nms_thres=0.3, conf_thres=0.3)  # j: (bboxes, confidences, det_indexes)  # j: YOLO-v3-lindernoren detector
                    detector_time += time.time() - start
                    # j: YOLO-80
                    # detections = detect.detect_image(self.detector, input_img, nms_thres=0.3, conf_thres=0.38)  # j: (bboxes, confidences, det_indexes)  # j: YOLO-v3-lindernoren detector

                    # j: xyxy (corners) --> xywh (center+size)
                    detections_xywh = np.empty([detections.shape[0], 4])
                    detections_xywh[:, 2] = w = detections[:, 2] - detections[:, 0]
                    detections_xywh[:, 3] = h = detections[:, 3] - detections[:, 1]
                    detections_xywh[:, 0] = xc = detections[:, 0] + (w / 2)
                    detections_xywh[:, 1] = yc = detections[:, 1] + (h / 2)

                    cls_conf = detections[:, 4]
                    cls_ids = detections[:, 5]

                    grid_x = ((xc / input_img.shape[1]) * scale[0]).astype(int)
                    grid_y = ((yc / input_img.shape[0]) * scale[1]).astype(int)

                    # draw detection origin cells
                    # cell_x = input_img.shape[1] / scale[0]
                    # cell_y = input_img.shape[0] / scale[1]
                    # for idx, (i,j) in enumerate(zip(grid_x, grid_y)):
                    #     # print(f"grid x: {i}, grid y: {j}")
                    #     pos_x = int(i * cell_x)
                    #     pos_y = int(j * cell_y)
                    #     pos_x2 = int(pos_x + cell_x)
                    #     pos_y2 = int(pos_y + cell_y)
                    #     cv2.rectangle(input_img, (pos_x, pos_y), (pos_x2, pos_y2), (0, 255, 0), 1)
                    #
                    #     # DEBUG: draw detection centers
                    #     c_x = int(xc[idx])
                    #     c_y = int(yc[idx])
                    #     cv2.rectangle(input_img, (c_x-1, c_y-1), (c_x+1, c_y+1), (255, 255, 0), 2)

                    # j: get reid targets
                    if scale == [13, 13]:
                        scale_idx = 0
                    if scale == [26, 26]:
                        scale_idx = 1
                    if scale == [52, 52]:
                        scale_idx = 2
                    reid_targets = reids[scale_idx][0, :, grid_y, grid_x]

                    # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
                    mask = cls_ids == 0
                    bbox_xywh = detections_xywh[mask]
                    cls_conf = cls_conf[mask]
                    reids_masked = reid_targets[mask]
                    detections_xywh[:, 3:] *= 1.2  # j: expands height and width

                    print(f"Confidences: {cls_conf}")
                ##################################################

                ########### YOLOv3 ################
                if not args.reid_yolo:
                    # do detection
                    start = time.time()
                    bbox_xywh, cls_conf, cls_ids = self.detector(input_img)
                    detector_time += time.time() - start

                    # select person class
                    mask = cls_ids == 0

                    bbox_xywh = bbox_xywh[mask]
                    # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
                    bbox_xywh[:, 3:] *= 1.2
                    cls_conf = cls_conf[mask]
                ##################################

                # do tracking
                start = time.time()
                if args.reid_yolo:
                    if not args.standalone_reid:
                        outputs = self.deepsort.update(bbox_xywh, cls_conf, input_img, reids_masked, standalone_reid=False)
                    else:
                        outputs = self.deepsort.update(bbox_xywh, cls_conf, input_img, standalone_reid=True)
                else:
                    if args.standalone_reid:
                        outputs = self.deepsort.update(bbox_xywh, cls_conf, input_img, standalone_reid=True)
                    else:
                        raise Exception("Not possible to have no reid_yolo and no standalone - there are no reid features!")
                tracker_time += time.time() - start

                # draw detections for visualization
                if len(outputs) > 0:

                    bbox_tlwh = []
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    input_img = draw_boxes(input_img, bbox_xyxy, identities)

                    # scale padded image back to original size
                    h_old = ori_im.shape[0]
                    w_old = ori_im.shape[1]
                    bbox_xyxy[:, 0], bbox_xyxy[:, 1] = self.deepsort.padded_to_original_size(bbox_xyxy[:, 0],
                                                                                             bbox_xyxy[:, 1], h_old, w_old,
                                                                                             w_new, h_new)
                    bbox_xyxy[:, 2], bbox_xyxy[:, 3] = self.deepsort.padded_to_original_size(bbox_xyxy[:, 2],
                                                                                             bbox_xyxy[:, 3], h_old, w_old,
                                                                                             w_new, h_new)

                    for bb_xyxy in bbox_xyxy:
                        bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                    results.append((idx_frame - 1, bbox_tlwh, identities))

                if self.args.display:
                    cv2.imshow("test", input_img)
                    cv2.waitKey(1)

                if self.args.save_path:
                    # TODO: convert to cmd parameter (save video or files)
                    self.writer.write(input_img)  # j: save as video frames
                    cv2.imwrite(self.args.save_path + f"/{str(idx_frame).zfill(3)}_scale{scale}.jpeg", input_img)  # j: save as separate images
                    pass

                # save results
                write_results(self.save_results_path, results, 'mot')

                # logging
                # self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                #                  .format(end - start, 1 / (end - start), bbox_xywh.shape[0], len(outputs)))
                n_detections += bbox_xywh.shape[0]
                n_tracks += len(outputs)

            self.vdo.set(cv2.CAP_PROP_POS_FRAMES, 0)  # j: reset video
            avg_time = detector_time / idx_frame
            avg_tracker_time = tracker_time / idx_frame
            avg_detections = n_detections / idx_frame
            avg_tracks = n_tracks / idx_frame
            avg_nms_time = self.deepsort.nms_time / idx_frame
            avg_reid_standalone_time = self.deepsort.reid_standalone_time / idx_frame
            print(f"Scale {scale}:")
            print(f"Avg detector time: {avg_time:.3f}")
            print(f"Avg tracker time: {avg_tracker_time:.3f}")
            print(f"Avg detections: {avg_detections:.3f}")
            print(f"Avg tracks: {avg_tracks:.3f}")
            print(f"Avg nms time: {avg_nms_time}")
            print(f"Avg standalone ReID time (out of tracker time): {avg_reid_standalone_time if args.standalone_reid else 'not used'}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_detection", type=str, default="/home/julius-think/Thesis/Code/deep_sort_pytorch/configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="/home/julius-think/Thesis/Code/deep_sort_pytorch/configs/deep_sort.yaml")
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    parser.add_argument("--reid_yolo", type=int)
    parser.add_argument("--standalone_reid", type=int)

    # YOLO-v3 lindernoren args

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.save_path = os.path.dirname(__file__) + "/" + args.save_path  # j: update to relative path
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    with VideoTracker(cfg, args, video_path=os.path.dirname(__file__) + "/" + args.VIDEO_PATH) as vdo_trk:
        vdo_trk.run()
