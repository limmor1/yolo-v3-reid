# TODO: you may need to adjust the dataset path logic

import detector_YOLO_v3_REID.YOLOv3_lindernoren.train
import argparse, os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO model.")
    parser.add_argument("-m", "--model", type=str, help="Path to model definition file (.cfg)")
    # parser.add_argument("-d", "--data", type=str, default="config/coco.data", help="Path to data config file (.data)")
    parser.add_argument("-d", "--data", type=str, help="Path to data config file (.data)")
    parser.add_argument("-e", "--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("-v", "--verbose", action='store_true', help="Makes the training more verbose")
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--pretrained_weights", type=str, help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="Interval of epochs between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, help="Interval of epochs between evaluations on validation set")
    parser.add_argument("--multiscale_training", action="store_true", help="Allow for multi-scale training")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="Evaluation: IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.1, help="Evaluation: Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="Evaluation: IOU threshold for non-maximum suppression")
    parser.add_argument("--logdir", type=str, default="logs", help="Directory for training log files (e.g. for TensorBoard)")
    parser.add_argument("--cutoff", type=int, default=None, help="Number of modules to load from the .weights file (e.g. Darknet-53 only)")  #j: cutoff for module weight loading
    parser.add_argument("--start_epoch", type=int, default=None, help="Starting epoch (used when resuming for correct stats naming")
    parser.add_argument("--no_reid_end", type=bool, default=False, help="Do not load classifier head (solves weakly supervised --> strongly supervised head size mismatch)")

    args = parser.parse_args()

    args.data = os.path.dirname(__file__) + "/" + args.data
    args.model = os.path.dirname(__file__) + "/" + args.model
    args.pretrained_weights = os.path.dirname(__file__) + "/" + args.pretrained_weights

    print(args)
    detector_YOLO_v3_REID.YOLOv3_lindernoren.train.train_yolov3_reid(args)
    
