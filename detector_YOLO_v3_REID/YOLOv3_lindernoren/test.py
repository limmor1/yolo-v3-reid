#! /usr/bin/env python3

from __future__ import division

from detector_YOLO_v3_REID.YOLOv3_lindernoren.models import *
from detector_YOLO_v3_REID.YOLOv3_lindernoren.utils.utils import *
from detector_YOLO_v3_REID.YOLOv3_lindernoren.utils.datasets import *
from detector_YOLO_v3_REID.YOLOv3_lindernoren.utils.augmentations import *
from detector_YOLO_v3_REID.YOLOv3_lindernoren.utils.transforms import *
from detector_YOLO_v3_REID.YOLOv3_lindernoren.utils.parse_config import *

import os
import argparse
import tqdm

from terminaltables import AsciiTable

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable


def evaluate_model_file(model_path, weights_path, img_path, class_names,
    batch_size=8, img_size=416, n_cpu=8,
    iou_thres=0.5, conf_thres=0.5, nms_thres=0.5, verbose=True):
    """Evaluate model on validation dataset.

    :param model_path: Path to model definition file (.cfg)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :param img_path: Path to file containing all paths to validation images.
    :type img_path: str
    :param class_names: List of class names
    :type class_names: [str]
    :param batch_size: Size of each image batch, defaults to 8
    :type batch_size: int, optional
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param n_cpu: Number of cpu threads to use during batch generation, defaults to 8
    :type n_cpu: int, optional
    :param iou_thres: IOU threshold required to qualify as detected, defaults to 0.5
    :type iou_thres: float, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :param verbose: If True, prints stats of model, defaults to True
    :type verbose: bool, optional
    :return: Returns precision, recall, AP, f1, ap_class
    """
    dataloader = _create_validation_data_loader(img_path, batch_size, img_size, n_cpu)
    model = load_model(model_path, weights_path)
    metrics_output = _evaluate(
        model,
        dataloader,
        class_names,
        img_size,
        iou_thres,
        conf_thres,
        nms_thres,
        verbose)
    return metrics_output

def print_eval_stats(metrics_output, class_names, verbose):
    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output

        if verbose:
            # Prints class AP and mean AP
            ap_table = [["Index", "Class", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
        print(f"---- mAP {AP.mean():.5f} ----")
    else:
        print( "---- mAP not measured (no detections found by model) ----")

def _evaluate(model, dataloader, class_names, img_size, iou_thres, conf_thres, nms_thres, verbose):
    """Evaluate model on validation dataset.

    :param model: Model to evaluate
    :type model: models.Darknet
    :param dataloader: Dataloader provides the batches of images with targets
    :type dataloader: DataLoader
    :param class_names: List of class names
    :type class_names: [str]
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param iou_thres: IOU threshold required to qualify as detected
    :type iou_thres: float
    :param conf_thres: Object confidence threshold
    :type conf_thres: float
    :param nms_thres: IOU threshold for non-maximum suppression
    :type nms_thres: float
    :param verbose: If True, prints stats of model
    :type verbose: bool
    :return: Returns precision, recall, AP, f1, ap_class
    """
    model.eval()  # Set model to evaluation mode

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for _, imgs, targets in tqdm.tqdm(dataloader, desc="Validating"):
        # j: targets: N x tensor(img, class_id, box_id, x1, y1, x2, y2)
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        # targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 3:] = xywh2xyxy(targets[:, 3:])  # j: move indexes to accommodate unique ID
        # targets[:, 2:] *= img_size  # j: move indexes to accommodate unique ID
        targets[:, 3:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        predictions, _ = model(imgs)
        with torch.no_grad():
            outputs = to_cpu(predictions)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, iou_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets[:, [i for i in range(targets.shape[1]) if i != 2]], iou_threshold=iou_thres)
    
    if len(sample_metrics) == 0:  # No detections over whole validation set.
        print("---- No detections over whole validation set ----")
        return None
    
    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    metrics_output = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    print_eval_stats(metrics_output, class_names, verbose)

    return metrics_output

def _create_validation_data_loader(img_path, batch_size, img_size, n_cpu):
    """Creates a DataLoader for validation.

    :param img_path: Path to file containing all paths to validation images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = ListDataset(img_path, img_size=img_size, multiscale=False, transform=DEFAULT_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn)
    return dataloader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate validation data.")
    # parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg", help="Path to model definition file (.cfg)")
    parser.add_argument("-m", "--model", type=str, help="Path to model definition file (.cfg)")
    parser.add_argument("-w", "--weights", type=str, help="Path to weights or checkpoint file (.weights or .pth)")
    parser.add_argument("-d", "--data", type=str, help="Path to data config file (.data)")
    parser.add_argument("-b", "--batch_size", type=int, help="Size of each image batch")
    parser.add_argument("-v", "--verbose", action='store_true', help="Makes the validation more verbose")
    parser.add_argument("--img_size", type=int, default=416, help="Size of each image dimension for yolo")
    parser.add_argument("--n_cpu", type=int, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--iou_thres", type=float, help="IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, help="Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, help="IOU threshold for non-maximum suppression")
    # j: n.b. testing IoU confidence threshold is much higher than training (0.5 > 0.1)
    # TODO: why is the conf_thres set to 0.5?

    args = parser.parse_args()
    # args.n_cpu = 0  # TODO: DEBUG
    print(args)

    # Load configuration from data file
    data_config = parse_data_config(args.data)
    valid_path = data_config["valid"]  # Path to file containing all images for validation
    class_names = load_classes(data_config["names"])  # List of class names

    precision, recall, AP, f1, ap_class = evaluate_model_file(
        args.model,
        args.weights,
        valid_path,
        class_names,
        batch_size=args.batch_size,
        img_size=args.img_size,
        n_cpu=args.n_cpu,
        iou_thres=args.iou_thres,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        verbose=True)
    print(f"precision: {precision}, \nrecall: {recall}, \nAP: {AP}, \nf1: {f1}, \nap_class: {ap_class}")
