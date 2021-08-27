#  Qualitative tests of validation loss from the loss functions
#  Loads each trained checkpoint and runs loss on validation dataset

#  Julius Raskevicius, 2021, VUB
#  PyTorch implementation of Yolov3 by Linder Noren

import numpy as np
from models import *
from utils.transforms import *
import argparse
from utils.datasets import *
from torch.utils.data import DataLoader
import tqdm
from utils.loss import compute_loss
from terminaltables import AsciiTable
from utils.logger import *

np.set_printoptions(suppress=True)
if __name__ == "__main__":
    print("Performing validation loss test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # parse cmd args
    parser = argparse.ArgumentParser(description="Evaluate validation data.")
    parser.add_argument("-m", "--model", type=str, help="Path to model definition file (.cfg)")
    parser.add_argument("--n_cpu", type=int, default=6, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--logdir", type=str, default="logs", help="Directory for validation log files (e.g. for TensorBoard)")
    parser.add_argument("--start_epoch", type=int, help="Starting epoch")
    parser.add_argument("--end_epoch", type=int, help="Ending epoch")
    parser.add_argument("--img_size", type=int, default=416, help="Size of each image dimension for yolo")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="Interval of epochs between evaluations on validation set")
    parser.add_argument("--cutoff", type=int, default=None, help="Number of modules to load from the .weights file (e.g. Darknet-53 only)")  # cutoff for module weight loading
    parser.add_argument("--no_reid_end", type=bool, default=False, help="Number of modules to load from the .weights file (e.g. Darknet-53 only)")  # cutoff for module weight loading
    parser.add_argument("-d", "--data", type=str, help="Path to data config file (.data)")
    parser.add_argument("--minibatch", type=int, help="Number of images in a validation minibatch")
    args = parser.parse_args()
    print(args)

    # init logger
    logger = Logger(args.logdir)  # Tensorboard logger

    # load cfg file
    data_config = parse_data_config(args.data)
    img_path = data_config["valid"]
    mini_batch_size = args.minibatch

    dataset = ListDataset(img_path, img_size=args.img_size, multiscale=False, transform=DEFAULT_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=mini_batch_size,
        shuffle=False,
        num_workers=args.n_cpu,
        # num_workers=0,
        pin_memory=True,
        collate_fn=dataset.collate_fn)

    # perform validations on each checkpoint
    for i in range(args.start_epoch, args.end_epoch+1, args.evaluation_interval):
        print(f"Epoch {i}:")
        print("--------------------------")
        # load and init model
        pretrained_weights = f"checkpoints/yolov3_ckpt_{i}.pth"
        model = load_model(args.model, pretrained_weights, args.cutoff, args.no_reid_end)
        n_obj = model.hyperparams["n_obj_ids"]  #  number of unique objects in dataset (!= n_classes)

        # set model to validation mode
        # we need to do this, because we need the model to output the same format as in training (but not be set to training)
        for yolo_layer in model.yolo_layers:  
            yolo_layer.validation = True
        for reid_layer in model.reid_layers:
                reid_layer.validation = True
        model.validation = True
        model.eval()

        # calculate loss on each image in the dataset
        tot_loss = 0
        tot_loss_components = torch.zeros(5, requires_grad=False)
        for idx, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Validating simple loss")):
            # cv2.imwrite(f"valid_input_checkpoint{i}_{j}.png", idx)  #  preview input

            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device)

            outputs, _ = model(imgs)

            loss, loss_components = compute_loss(outputs, None, targets, model, n_obj)
            loss = float(loss)
            loss /= len(dataloader)
            tot_loss += loss
            loss_components /= len(dataloader)
            tot_loss_components += loss_components

        # Log
        iou_loss, obj_loss, cls_loss, tot_loss, reid_loss = tot_loss_components
        iou_loss = float(iou_loss)
        obj_loss = float(obj_loss)
        cls_loss = float(cls_loss)
        tot_loss = float(tot_loss)
        reid_loss = float(reid_loss)

        log_str = ""
        log_str += AsciiTable(
            [
                ["Type", "Value"],
                ["IoU loss", iou_loss],
                ["Object loss", obj_loss],
                ["Class loss", cls_loss],
                # ["REID loss", reid_loss],
                ["Total loss", tot_loss],
            ]).table
        print("\n" + log_str)

        tensorboard_log = [
            ("validate/iou_loss", iou_loss),
            ("validate/obj_loss", obj_loss),
            ("validate/class_loss", cls_loss),
            # ("validate/reid_loss", reid_loss),
            ("validate/total_loss", tot_loss)]
        logger.list_of_scalars_summary(tensorboard_log, i)

        del imgs
        del outputs
        del loss
        del loss_components
        del tot_loss
        del tot_loss_components
