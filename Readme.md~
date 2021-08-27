# About #
The goal of this project was to integrate the standalone ReID feature embedding module into the YOLO-v3. Visual feature embeddings are useful in pedestrian tracking as they allow distinguishing bounding boxes based on appearance. The new signal allows resolving occlusions or missing detections. DeepSORT uses a stand alone module that incurs additional computation cost. In contrast, YOLO-v3-REID is trained to perform the same task but at no extra cost. In training FairMOT cost function is used (Yifu Zhang, Chunyu Wang, Xinggang Wang, Wenjun Zeng, & Wenyu Liu. (2020). FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking).

# Setup # 
* `$ conda create -n yolo-v3-reid`
* `$ conda install pip`
* `$ pip install -r requirements.txt`
* `$ conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge`
* download weights for each model and place them in the right paths (see table below)

## Weights ##
Checkpoints with saved weights for each model can be downloaded from these locations:

Weight path | Weights location in Google Drive
------------|-----------------------------------
Yolo-v3-REID/deep_sort/deep/checkpoint | [TODO: link] weights/deepsort
Yolo-v3-REID/detector_YOLO_v3/YOLOv3/weight |  [TODO: link] weights/detector_yolov3_zqpei
Yolo-v3-REID/detector_YOLO_v3_REID/YOLOv3_lindernoren/weights|  [TODO: link] weights/detector_yolov3_reid_ours_transfer
Yolo-v3-REID/detector_YOLO_v3_REID/YOLOv3_lindernoren/checkpoints |  [TODO: link] weights/detector_yolov3_reid_ours_model

# Usage #
Run yolov3_reid_deepsort.py to test the model. Place the input video in `/examples`. The result is saved as images to `/output`
* `$ python yolov3_reid_deepsort.py examples/pedestrian_input.avi --save_path output --reid_yolo 1 --standalone_reid 1 --display`
To run on CPU use --cpu flag.

## ReID module options ##
Integrating the re-identification module into the detector can affect detector performance. To show that this effect is minimal you can do detections with original Yolo-v3 network (Option 3) and compare with our model (Option 2).

 Configuration                     | Effect
-----------------------------------|-----------------------------
 --reid_yolo 0 --standalone_reid 1 | use a standalone ReID module (original Z. Pei)
 --reid_yolo 1 --standalone_reid 0 | use integrate ReID module (our modification)
 --reid_yolo 1 --standalone_reid 1 | test effect of integration on detections (detections Yolo-v3 Z. Pei + ReID with our features)

# Performance #
## Tracker time ##
The diagram shows performance improvement of YOLO-v3-REID over YOLO-v3 + standalone DeepSORT feature extractor network. 3 configurations are tested, with "skip-end" and "mix-end" model showing overall improvement in time.
[TODO: DIAGRAM]

## Tracker performance ##
MOT16 benchmark was used to evaluate overall tracker performance. The table shows improvements in MOTA score and mostly tracked/lost track as well as identity switches over a bypass condition (no ReID signal used).
[TODO: DIAGRAM]

## ReID performance ##
YOLO-v3 produces detections at three scales. For each scale YOLO-v3-REID outputs a feature embedding vector. ROC curve for all three scales is shown below to illustrate the discerning power of the ReID output. Diagonal illustrates a no-skill model.
[TODO: DIAGRAM]

# Training #
Training is done on a combined dataset from MOT benchmarks (MOT15 + MOT16 + MOT20) with some sequences removed for validation.
* `$ python yolov3_reid_train.py --multiscale --n_cpu 4 --data "detector_YOLO_v3_REID/YOLO-lindernoren/config/mot_combined.data" --model "detector_YOLO_v3_REID/YOLO-lindernoren/config/yolov3_80class_reid_allscale_MOTcombined_nosoft_mixed_end.cfg" --pretrained_weights "detector_YOLO_v3_REID/YOLO-lindernoren/weights/yolov3.weights" --start_epoch 0 --evaluation_interval 999 --cutoff 107 --verbose`

## Training flags ##
flag | meaning
-----|--------
--multiscale | variate input image scale to train-in resiliance to different input sizes
--n_cpu <INT> | number of CPUs used in dataloader
--data <PATH> | path to training dataset config file
--model <PATH> | path to model
--pretrained_weights <PATH> | path to YOLO-v3 model weights
--start_epoch <INT> | when resuming set this to != 0
--evaluation_interval <INT> | run validation of the detector every n epochs
--cutoff <INT>  | how many layers to take from loaded weights when training the model (107 = initialization with detector heads, 75 = without)
--verbose  | see extra output

## Dataset structure ##
`--data` flag defines the path to a dataset configuration file that contains paths to files listing images in training and validation datasets. Each of the `.train` and `.val` files should contain the list of paths to each training image.
 
Training data should be placed inside `/Yolo-v3-REID/detector_YOLO_v3_REID/YOLOv3_lindernoren/data/` with the following folder structure:
<FOLDERS>. The dataset is accessible from [TODO: google drive link for the dataset]
You also need to create a symbolic link `Yolo-v3-REID/data/data --> Yolo-v3-REID/detector_YOLO_v3_REID/YOLOv3_lindernoren/data`

# Credit #
PyTorch implementation of DeepSORT using YOLO-v3 by Z. Pei (https://github.com/ZQPei/deep_sort_pytorch)
PyTorch YOLO-v3 implementation by E. Lindernoren (https://github.com/eriklindernoren/PyTorch-YOLOv3)
Original YOLO-v3 implementation by J. Redmon (Joseph Redmon, & Ali Farhadi. (2018). YOLOv3: An Incremental Improvement)
