#! /usr/bin/env python3

from __future__ import division

import os
import argparse
import tqdm

from detector_YOLO_v3_REID.YOLOv3_lindernoren.models import *
from detector_YOLO_v3_REID.YOLOv3_lindernoren.utils.logger import *
from detector_YOLO_v3_REID.YOLOv3_lindernoren.utils.utils import *
from detector_YOLO_v3_REID.YOLOv3_lindernoren.utils.datasets import *
from detector_YOLO_v3_REID.YOLOv3_lindernoren.utils.augmentations import *
from detector_YOLO_v3_REID.YOLOv3_lindernoren.utils.transforms import *
from detector_YOLO_v3_REID.YOLOv3_lindernoren.utils.parse_config import *
from detector_YOLO_v3_REID.YOLOv3_lindernoren.utils.loss import compute_loss
from detector_YOLO_v3_REID.YOLOv3_lindernoren.test import _evaluate, _create_validation_data_loader

from terminaltables import AsciiTable

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from torchsummary import summary

# j: imports
import gc


def _create_data_loader(img_path, batch_size, img_size, n_cpu, args):
    """Creates a DataLoader for training.

    :param img_path: Path to file containing all paths to training images.
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
    dataset = ListDataset(
        img_path,
        img_size=img_size,
        multiscale=args.multiscale_training,
        transform=AUGMENTATION_TRANSFORMS)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        # drop_last=True
    )
    return dataloader


def train_yolov3_reid(args):
    with open("output/experiment_args", "w") as f:
        f.write(str(args))

    logger = Logger(args.logdir)  # Tensorboard logger

    # Create output directories if missing
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(args.data)
    train_path = os.path.dirname(__file__) + "/" + data_config["train"]
    valid_path = os.path.dirname(__file__) + "/" + data_config["valid"]
    class_names = load_classes(data_config["names"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ############
    # Create model
    # ############

    cutoff = args.cutoff
    start_epoch = args.start_epoch
    model = load_model(args.model, args.pretrained_weights, cutoff, args.no_reid_end)
    n_obj = model.hyperparams["n_obj_ids"]  # j: number of unique objects in dataset (!= n_classes)
    # Print model
    if args.verbose:
        summary(model, input_size=(3, model.hyperparams['height'], model.hyperparams['height']))

    mini_batch_size = model.hyperparams['batch'] // model.hyperparams['subdivisions']
    # #################
    # Create Dataloader
    # #################

    # Load training dataloader
    dataloader = _create_data_loader(  # j: very simple dataset, only reads images and labels from pregenerated file
        train_path,  #: j: path to a txt which list image paths
        mini_batch_size, 
        model.hyperparams['height'], 
        args.n_cpu, # 0)  # j: TODO: for debug
        args)


    # Load validation dataloader
    validation_dataloader = _create_validation_data_loader(
        valid_path, 
        mini_batch_size, 
        model.hyperparams['height'], 
        args.n_cpu)
        # 0)  # j: TODO: for debug

    # ################
    # Create optimizer
    # ################
    
    params = [p for p in model.parameters() if p.requires_grad]
    acc_n_gradients = int(model.hyperparams["grad_accumulation"])   # j: number of minibatches to accumulate loss for
    acc_loss = 0
    acc_loss_components = torch.zeros(5)
    if (model.hyperparams['optimizer'] in [None, "adam"]):  # j: use Adam optimizer in training
        optimizer = torch.optim.Adam(
            params, 
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
            )
    elif (model.hyperparams['optimizer'] == "sgd"):
        optimizer = torch.optim.SGD(
            params, 
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
            momentum=model.hyperparams['momentum'])
    else:
        print("Unknown optimizer. Please choose between (adam, sgd).")

    ################################
    # Start epochs & minibatch loops
    ################################
    for epoch in range(start_epoch, args.epochs):
        print("\n---- Training Model ----")
        model.train()  # Set model to training mode
        
        for batch_i, (img_path, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")):
            if img_path == None:  # j: prevent empty batch errors
                continue
            batch_size = imgs.shape[0]
            # batches_done = len(dataloader) * epoch + batch_i
            batches_done = int(len(dataloader) / acc_n_gradients) * epoch + ((batch_i + 1) / acc_n_gradients)    # j: adjust batches_done to gradient accumulation rate
            # print(f"imgs.shape {imgs.shape}; targets.shape {targets.shape}")
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device)

            outputs, reid_outputs = model(imgs)

            # TODO: debug
            # debug_img = imgs.squeeze(0).to("cpu").numpy().transpose([1,2,0])[:, :, ::-1]*255
            # cv2.imwrite("/home/julius-think/Desktop/input.jpg", debug_img)
            # outputs = model(imgs)

            # TODO: DEBUG: FairMOT loss "s" factor (move to a backup file)
            loss, loss_components = compute_loss(outputs, reid_outputs, targets, model, n_obj, model.s_det, model.s_id)
            # loss, loss_components = compute_loss(outputs, reid_outputs, targets, model, n_obj, model.s_det, model.s_id, debug_img)
            loss = loss / acc_n_gradients  # j: scale loss to our accumulation steps
            loss_components = loss_components / acc_n_gradients
            loss.backward()  # j: add to parameter gradients

            acc_loss += loss
            acc_loss_components += loss_components

            ###############
            # Run optimizer
            ###############

            if int(batches_done) % model.hyperparams['subdivisions'] == 0:  # j: batches_done is fractional since acc_grad addition, pretend its not with int conversion
                # Adapt learning rate
                # Get learning rate defined in cfg
                lr = model.hyperparams['learning_rate']
                if batches_done < model.hyperparams['burn_in']:
                    # Burn in
                    lr *= (batches_done / model.hyperparams['burn_in'])
                else:
                    # Set and parse the learning rate to the steps defined in the cfg
                    for threshold, value in model.hyperparams['lr_steps']:
                        if batches_done > threshold:
                            lr *= value
                # Log the learning rate
                logger.scalar_summary("train/learning_rate", lr, batches_done)
                # Set learning rate
                for g in optimizer.param_groups:
                        g['lr'] = lr

                # Run optimizer j: with gradient accumulation
                if (batch_i + 1) % acc_n_gradients == 0:
                    optimizer.step()
                    # Reset gradients
                    optimizer.zero_grad()

            # j: cleanup
            iou_loss, obj_loss, cls_loss, tot_loss, reid_loss = acc_loss_components
            iou_loss = float(iou_loss)
            obj_loss = float(obj_loss)
            cls_loss = float(cls_loss)
            tot_loss = float(tot_loss)
            reid_loss = float(reid_loss)

            del reid_outputs
            del imgs
            del outputs
            del loss
            del loss_components

            ##############
            # Log progress
            ##############
            # j: n.b. slightly misleading,
            #    reported loss is never of the grad. accumulation sequence, just the last batch

            if (batch_i + 1) % acc_n_gradients == 0:
                log_str = ""
                log_str += AsciiTable(  # j: terminal tables package
                    [
                        ["Type", "Value"],
                        ["IoU loss", iou_loss],
                        ["Object loss", obj_loss],
                        ["Class loss", cls_loss],
                        ["REID loss", reid_loss],
                        ["Loss", tot_loss],
                    ]).table

                if args.verbose: print("\n" + log_str)

                # Tensorboard logging
                tensorboard_log = [
                        ("train/iou_loss", iou_loss),
                        ("train/obj_loss", obj_loss),
                        ("train/class_loss", cls_loss),
                        ("train/reid_loss", reid_loss),
                        ("train/loss", acc_loss),
                        ("train/s_det", model.s_det),
                        ("train/s_id", model.s_id)
                        ]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

                # # j: Reset accumulation losses
                acc_loss = 0
                acc_loss_components = torch.zeros(5)

            model.seen += batch_size

        # #############
        # Save progress
        # #############

        # Save model to checkpoint file
        if epoch % args.checkpoint_interval == 0:
            checkpoint_path = f"checkpoints/yolov3_ckpt_{epoch}.pth"
            print(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
            torch.save(model.state_dict(), checkpoint_path)

        # ########
        # Evaluate
        # ########
        # Evaluate precision / recall on validation set
        if (epoch + 1) % args.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            metrics_output = _evaluate(
                model,
                validation_dataloader,
                class_names,
                img_size=model.hyperparams['height'],
                iou_thres=args.iou_thres,
                conf_thres=args.conf_thres,
                nms_thres=args.nms_thres,
                verbose=args.verbose
            )

            if metrics_output is not None:
                precision, recall, AP, f1, ap_class = metrics_output
                evaluation_metrics = [
                    ("validation/precision", precision.mean()),
                    ("validation/recall", recall.mean()),
                    ("validation/mAP", AP.mean()),
                    ("validation/f1", f1.mean())]
                logger.list_of_scalars_summary(evaluation_metrics, epoch)