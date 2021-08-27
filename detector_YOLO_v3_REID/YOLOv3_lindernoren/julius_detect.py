# j: Qualitative tests of REID loss

# j: Julius Raskevicius, 2021, VUB
# j: PyTorch implementation of Yolov3 by Linder Noren

import numpy as np
np.set_printoptions(suppress=True)

import detect
from models import *
import cv2

from scipy.spatial.distance import cdist
from utils.transforms import *
import os
from matplotlib import pyplot as plt
import math

from  matplotlib.colors import LinearSegmentedColormap

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def compare_reid_map_to_anchor(anchor_target, reid_map):
    """@params:
    anchor_target: features of a pixel that corresponds to the anchor
    reid_map: tensor(1, 128, 13, 13), network output for reid at one scale
    """
    scale_x, scale_y = reid_map.shape[2: 4]  # 13 or 26 or 52
    # reid_map = reid_maps[i].reshape(128, scale_x, scale_y)  # TODO: debug uncomment
    # reid_map = reid_map.reshape(64, scale_x, scale_y)
    reid_map = reid_map.squeeze(0).reshape(128, -1).transpose([1, 0])  # TODO: debug uncomment
    # reid_map = reid_map.reshape(64, -1).transpose([1,0])

    dist = cdist(reid_map, anchor_target, 'cosine') / 2  # j: [0; 1]
    dist = 1 * (1 - 8**(-dist/ dist.mean()))  # exponential mapping, same range [0; 1]
    # dist = cdist(reid_map, anchor_target, 'cosine')  # j: [0; 1]
    # dist = np.maximum(0.0, cdist(reid_map, anchor, 'cosine')) + off  # j: [0]
    # dist = (np.log(dist) - np.log(off)) / (np.log(1) - np.log(off))
    # f(x) = (log(x) - log(min)) / (log(max) - log(min)) # log mapping 0-1 range

    dist = dist.reshape(1, scale_x, scale_y)
    return dist

def draw_dist_map(img, dist):
    """@params:
    img - image
    dist - distance matrix (0, (scale_y, scale_x)"""
    alpha = 0.75
    cmap = LinearSegmentedColormap.from_list('rg',["r", "brown", "g"], N=256)
    scale_y, scale_x = dist.shape[1:3]
    overlay = img.copy()
    img_h = img.shape[0]
    img_w = img.shape[1]
    width = int(img_w / scale_y)
    height = int(img_h / scale_x)

    for i in range(scale_y):  # height
        for j in range(scale_x):  # width
            dist_ij = dist[0, i, j]
            x = int(j * width)
            y = int(i * height)
            color = np.array(cmap((1-dist_ij))[:-1])*255
            cv2.rectangle(overlay, (x, y), (x+width, y+height), (color[2], color[1], color[0]), -1)
    img_new = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return img_new

if __name__ == "__main__":
    # initialize options
    os.makedirs("output/reid_res", exist_ok=True)

    print("Performing REID qualitative validation tests")
    nms_thresh = 0.4
    conf_thresh = 0.4
    n = 3  # j: top n matching cells to show
    print(f"Options: nms threshold: {nms_thresh}, confidence threshold: {conf_thresh}")
    scales = [(13, 13), (26, 26), (52, 52)]
    use_cuda = 1

    # image_names = [
    #     "/home/julius-think/Thesis/Code/YOLO-lindernoren/data/mot15/images/reid_tests/train/train_anchor1.jpg",
    #     "/home/julius-think/Thesis/Code/YOLO-lindernoren/data/mot15/images/reid_tests/train/train_target1.jpg"
    # ]  # from MOT15 train

    # image_names = [
    #     "/home/julius-think/Thesis/Code/YOLO-lindernoren/data/mot15/images/reid_tests/test/test_anchor1.jpg",
    #     "/home/julius-think/Thesis/Code/YOLO-lindernoren/data/mot15/images/reid_tests/test/test_target1_1.jpg"
    # ]  # from MOT15 test

    # image_names = [
    #     "/home/julius-think/Thesis/Code/YOLO-lindernoren/data/mot15/images/reid_tests/test/test_anchor1.jpg",
    #     "/home/julius-think/Thesis/Code/YOLO-lindernoren/data/mot15/images/reid_tests/test/test_target2_2.jpg"
    # ]  # from MOT15 test

    image_names = [
        "/home/julius-think/Thesis/Code/YOLO-lindernoren/data/mot15/images/reid_tests/test/test_anchor2.jpg",
        "/home/julius-think/Thesis/Code/YOLO-lindernoren/data/mot15/images/reid_tests/test/test_target2_1.jpg"
       ]  # from MOT20 test

    # Load model
    model_cfg_path = "/home/julius-think/Thesis/Code/YOLO-lindernoren/config/yolov3_80class_reid_allscale_MOTcombined_nosoft_mixed_end.cfg"

    # Load checkpoints
    # checkpoint_names = [f"checkpoints/yolov3_ckpt_{i}.pth" for i in range(0, 28+1, 1)]
    checkpoint_names = ["checkpoints/yolov3_ckpt_0.pth"]

    # Load anchor and target images
    img_anchor_ = cv2.imread(image_names[0])
    img_target_ = cv2.imread(image_names[1])

    # resize and pad image to 416x416
    img_anchor_, _ = transforms.Compose([
        transforms.Compose([
            AbsoluteLabels(),
            PadSquare(),
            RelativeLabels(),
            ToTensor(),
        ]),
        Resize(416)])(
        (img_anchor_, np.zeros((1, 6))))  # j: move indexes to accommodate unique ID
    img_anchor_ = np.array(transforms.ToPILImage()(img_anchor_))
    # cv2.imwrite(f"output/reid_res/input_anchor.png", img_anchor)  # j: preview input anchor

    # resize and pad image to 416x416
    img_target_, _ = transforms.Compose([
        transforms.Compose([
            AbsoluteLabels(),
            PadSquare(),
            RelativeLabels(),
            ToTensor(),
        ]),
        Resize(416)])(
        (img_target_, np.zeros((1, 6))))  # j: move indexes to accommodate unique ID
    img_target_ = np.array(transforms.ToPILImage()(img_target_))
    cv2.imwrite(f"output/reid_res/input_target.png", img_target_)  # j: preview input target


    # RUN tests
    min_confidences = []
    max_confidences = []
    for i_chk, checkpoint in enumerate(checkpoint_names):
        print(f"Checkpoint {i_chk}:")
        print("--------------------------")

        model = load_model(model_cfg_path, checkpoint)

        # j: YOLO-REID
        anchor_detections, reids_anchor = detect.detect_image(model, img_anchor_, nms_thres=nms_thresh, conf_thres=conf_thresh)
        target_detections, reids_target = detect.detect_image(model, img_target_, nms_thres=nms_thresh, conf_thres=conf_thresh)

        for obj_id, det in enumerate(anchor_detections):
            img_anchor = img_anchor_.copy()
            print(f"Testing object {obj_id} as target...")

            # get detection stats
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            c_x = x1 + int((x2 - x1) / 2)
            c_y = y1 + int((y2 - y1) / 2)

            # draw bbox on anchor
            if cls == 0:  # j: only display people
                cv2.rectangle(img_anchor, (x1, y1), (x2, y2), (0, 255, 0))

            for scale_idx in range(3):
                # get grid cell for the object center
                grid_x = int(c_x / 416 * scales[scale_idx][0])
                grid_y = int(c_y / 416 * scales[scale_idx][1])

                # draw anchors on anchor image (for every scale)
                x = int(grid_x * 416 / scales[scale_idx][0])
                y = int(grid_y * 416 / scales[scale_idx][1])
                x2 = int(x + 416 / scales[scale_idx][0])
                y2 = int(y + 416 / scales[scale_idx][1])
                cv2.rectangle(img_anchor, (x, y), (x2, y2), (255, 0, 0), 1)

            # save anchor image
            cv2.imwrite(f"output/reid_res/anchors_obj{obj_id}_epoch{i_chk}.jpg", img_anchor)


            # GENERATE TARGET IMAGE
            for scale_idx in range(3):
                img_target = img_target_.copy()

                # get grid cell for the object center
                grid_x = int(c_x / 416 * scales[scale_idx][0])
                grid_y = int(c_y / 416 * scales[scale_idx][1])

                # get anchor features for that detection
                anchor_features = reids_anchor[scale_idx][0, :, grid_y, grid_x].reshape(-1, 1).transpose([1, 0])

                # calculate heatmap
                dists = compare_reid_map_to_anchor(anchor_features, reids_target[scale_idx])
                img_target = draw_dist_map(img_target, dists)

                # j: mark top n cells on target image
                # top_n = [np.unravel_index(dists.argmin(), dists.shape)[1:]]  # min index
                top_n = list(zip(*np.unravel_index(dists.reshape(-1).argsort()[:n], dists.shape)[1:])) # n min indexes
                for top_idx, (top_i, top_j) in enumerate(top_n):
                    x = int(top_j * 416 / scales[scale_idx][0])
                    y = int(top_i * 416 / scales[scale_idx][1])
                    x2 = int(x + 416 / scales[scale_idx][0])
                    y2 = int(y + 416 / scales[scale_idx][1])
                    cv2.rectangle(img_target, (x, y), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(img_target, str(top_idx), (x, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))

                # mark target image detection centers
                for target_det in target_detections:
                    t_x1, t_y1, t_x2, t_y2, t_conf, t_cls = target_det
                    t_x1, t_y1, t_x2, t_y2 = int(t_x1), int(t_y1), int(t_x2), int(t_y2)
                    t_c_x = t_x1 + int((t_x2 - t_x1) / 2)
                    t_c_y = t_y1 + int((t_y2 - t_y1) / 2)
                    cv2.rectangle(img_target, (t_c_x - 1, t_c_y - 1), (t_c_x + 1, t_c_y + 1), (0, 255, 255), 2)

                # save target heatmap
                cv2.imwrite(f"output/reid_res/target_obj{obj_id}_epoch{i_chk}_scale{scale_idx}.jpg", img_target)


        # calc stats
        if len(anchor_detections) != 0:
            min_confidence = np.min(anchor_detections[:, 4], 0)
            max_confidence = np.max(anchor_detections[:, 4], 0)
        else:
            min_confidence = -1
            max_confidence = -1

        print(f"    total detections: {len(anchor_detections)}")
        print(f"    min confidence: {min_confidence:.2g}")
        print(f"    max confidence: {max_confidence:.2g}")

        # gather total epoch stats
        min_confidences.append(min_confidence)
        max_confidences.append(max_confidence)

    plt.figure()
    plt.title("Min/Max detection confidences, exp ?")
    plt.xlabel("Epoch")
    plt.ylabel("Detection confidence")
    plt.plot(min_confidences)
    plt.plot(max_confidences)
    # plt.show()

