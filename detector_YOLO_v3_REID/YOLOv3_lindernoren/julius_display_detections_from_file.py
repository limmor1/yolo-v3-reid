# Test PyTorch implementation of Yolov3 by Linder Noren

# Julius Raskevicius, 2021, VUB
# PyTorch implementation of Yolov3 by Linder Noren

import numpy as np
np.set_printoptions(suppress=True)

import detect
from models import *
import cv2

from scipy.spatial.distance import cdist
from utils.transforms import *

def compare_reid_map_to_anchor(anchor_grid_xy, reid_map, scale):
    """@params:
    anchor_grid_xy:  (x, y) tuple, grid coordinates of an anchor in the image (e.g. (10,5))
    reid_map: tensor(bs, 128, 13, 13), network output for reid
    """
    scale_x, scale_y = scale
    gy, gx = anchor_grid_xy
    anchor = reid_map[0, :, gx, gy].reshape(-1, 1).transpose([1,0])
    reid_map = reid_map.reshape(128, scale_x, scale_y)
    reid_map = reid_map.reshape(128, -1).transpose([1,0])

    # dist = cdist(reid_map, anchor, 'cosine')
    dist = cdist(reid_map, anchor, 'cosine')
    dist = dist.reshape(1, scale_x, scale_y)
    return dist

def draw_dist_map(img, dist, scale):
    """@params:
    img - image
    dist - distance matrix (0, (scale_y, scale_x)
    scale - (scale_y, scale_x) (e.g. 13x13)"""
    alpha = 0.8
    scale_y, scale_x = scale
    overlay = img.copy()
    img_h = img.shape[0]
    img_w = img.shape[1]
    for i in range(scale_y):  # height
        for j in range(scale_x):  # width
            dist_ij = dist[0, i, j]
            width = int(img_w / scale_y)
            height = int(img_h / scale_x)
            x = int(j * width)
            y = int(i * height)
            cv2.rectangle(overlay, (x, y), (x+width, y+height), (0, 0, int(255*dist_ij)), -1)
    img_new = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return img_new

if __name__ == "__main__":
    # load model structure
    # model_cfg_path = "/home/julius-think/Thesis/Code/YOLO-lindernoren/config/yolov3.cfg"
    # model_cfg_path = "/home/julius-think/Thesis/Code/YOLO-lindernoren/config/yolov3_1class.cfg"
    # model_cfg_path = "/home/julius-think/Thesis/Code/YOLO-lindernoren/config/yolov3_1class_reid.cfg"

    # load weights
    # model_weights_path = "/home/julius-think/Thesis/Code/YOLO-lindernoren/weights/yolov3.weights"
    # model_weights_path = "checkpoints/yolov3_ckpt_26-resumed_overnight.pth"
    # model_weights_path = "checkpoints/yolov3_ckpt_25-no_resume.pth"
    # model_weights_path = "/home/julius-think/Thesis/Code/YOLO-lindernoren/exp5_1class_reid_resumed_from_exp4_mot15_no_weak_150epochs_13x13/checkpoints/yolov3_ckpt_135.pth"
    # model_weights_path = f"checkpoints/yolov3_ckpt_{i}.pth"

    # model = load_model(model_cfg_path, model_weights_path)

    # load image
    # img = cv2.imread("/home/julius-think/Thesis/Code/YOLO-lindernoren/data/crowdhuman/images/val/273271,1b9330008da38cd6_multi_target.jpg")
    # target = (3, 3)  # j: foreground large scale + background small scale
    img = cv2.imread("/home/julius-think/Thesis/Datasets/CenterTrack/data/mot17/test/MOT17-01-DPM/img1/000261_multi_target.jpg")
    # target = (7, 6)  # j: MOT17 test scene perspective
    # img = cv2.imread("/home/julius-think/Thesis/Datasets/CenterTrack/data/mot17/test/MOT17-03-DPM/img1/000001_multi_target.jpg")  # j: contains multiple instances of same object
    # target = (5, 4)  # j: MOT17 test scene top

    # scale = (13, 13)

    # resize and pad image to 416x416
    input_img, _ = transforms.Compose([
        DEFAULT_TRANSFORMS,
        Resize(416)])(
        (img, np.zeros((1, 6))))  # j: move indexes to accommodate unique ID
    input_img = np.array(transforms.ToPILImage()(input_img))
    # cv2.imwrite(f"result_input_{i}.png", input_img)  # j: preview padded input

    # j: YOLO-80
    # detections = detect.detect_image(model, input_img, nms_thres=0.1, conf_thres=0.1)
    # j: YOLO-REID
    # detections, reids = detect.detect_image(model, input_img, nms_thres=0.1, conf_thres=0.1)
    detections = torch.load("/home/julius-think/Thesis/Code/deep_sort_pytorch/output/detections.pth")
    # dist = compare_reid_map_to_anchor(target, reids, scale)
    # img_new = draw_dist_map(input_img, dist, scale)

    # j: YOLO-80
    # img_new = input_img
    print(detections)
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if cls == 0:  # j: only display people
            cv2.rectangle(input_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0))
            # cv2.rectangle(img_new, (x1, y1), (x2, y2), (0, 255, 0))
    cv2.imwrite(f"loaded_detections.png", input_img)
    # cv2.imwrite(f"result_all_ckpt_{i}.png", img_new)
