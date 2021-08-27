from __future__ import division
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from detector_YOLO_v3_REID.YOLOv3_lindernoren.utils.parse_config import *
from detector_YOLO_v3_REID.YOLOv3_lindernoren.utils.utils import to_cpu, non_max_suppression, weights_init_normal

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    hyperparams.update({
        'batch': int(hyperparams['batch']),
        'subdivisions': int(hyperparams['subdivisions']),
        'width': int(hyperparams['width']),
        'height': int(hyperparams['height']),
        'channels': int(hyperparams['channels']),
        'optimizer': hyperparams.get('optimizer'),
        'momentum': float(hyperparams['momentum']),
        'decay': float(hyperparams['decay']),
        'learning_rate': float(hyperparams['learning_rate']),
        'burn_in': int(hyperparams['burn_in']),
        'max_batches': int(hyperparams['max_batches']),
        'policy': hyperparams['policy'],
        'lr_steps': list(zip(map(int,   hyperparams["steps"].split(",")), 
                             map(float, hyperparams["scales"].split(","))))
    })
    assert hyperparams["height"] == hyperparams["width"], \
        "Height and width should be equal! Non square images are padded with zeros."
    output_filters = [hyperparams["channels"]]
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", nn.Sequential())

        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", nn.Sequential())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes)
            modules.add_module(f"yolo_{module_i}", yolo_layer)
        elif module_def["type"] == "reid":
            # j: reid module def consists of a ReIDLayer that maps input(...) --> output(bs, 13, 13, 128)
            has_classify = bool(int(module_def["classify_head"]))
            has_softmax = bool(int(module_def["softmax"]))
            in_size = int(module_def["in_size"])
            n_obj_ids = int(module_def["n_obj_ids"])
            hyperparams["n_obj_ids"] = n_obj_ids
            reid_layer = ReIDLayer(in_size, has_classify, n_obj_ids, has_softmax)
            modules.add_module(
                f"reid_{module_i}",
                reid_layer,
            )
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes):
        super(YOLOLayer, self).__init__()
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.no = num_classes + 5   # number of outputs per anchor
        self.no = num_classes + 5   # DEBUG tryout
        self.grid = torch.zeros(1)  # TODO

        anchors = torch.tensor(list(chain(*anchors))).float().view(-1, 2)
        self.register_buffer('anchors', anchors)
        self.register_buffer('anchor_grid', anchors.clone().view(1, -1, 1, 1, 2))
        self.stride = None
        self.validation = False

    def forward(self, x, img_size):
        stride = img_size // x.size(2)
        self.stride = stride
        bs, _, ny, nx = x.shape
        x = x.view(bs, self.num_anchors, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        # j: x(bs, 255, 13, 13) --x.view--> x(bs, num_anchors, 255, 13, 13)  # adding extra order with 3 dimensions to split the 255 content
        # j: this means that I can't reduce the 80-->1 class without changing the previous layer size in yolo*.cfg
        # x(bs,255,20,20) --> x(bs,3,20,20,85)

        # j: return output as dets x (5 + cls1 + ... + clsN) in inference
        # j: return output as tensor(mb, anch, 13, 13, cls1 + clsN) in training
        if not self.training and not self.validation:  # inference
            if self.grid.shape[2:4] != x.shape[2:4]:
                self.grid = self._make_grid(nx, ny).to(x.device)

            y = x.sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid.to(x.device)) * stride  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid  # wh
            y = y.view(bs, -1, self.no)

        return x if (self.training or self.validation) else y

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

class ReIDLayer(nn.Module):
    """ReID layer, outputs an embedding for each cell in the incoming layer"""

    def __init__(self, in_size, has_classify, n_obj_ids, has_softmax):
        """
        @params:
        size: int, embedded vector size
        n_obj_ids: int, number of classes to classify to (= number of unique objects in dataset)
        has_classify: boolean, add classification to n_obj_ids classes
        has_softmax: boolean, use softmax along classification dimension
        """
        super(ReIDLayer, self).__init__()
        self.has_classify = has_classify
        self.has_softmax = has_softmax
        self.n_obj_ids = n_obj_ids
        self.validation = False
        if has_classify:
            self.classify = nn.Linear(128, self.n_obj_ids)  # TODO: FairMOT loss, try and see if equivalent!
            # self.classify = nn.Sequential(
            #     nn.Conv2d(in_size, self.n_obj_ids, 1),
            #     nn.BatchNorm2d(n_obj_ids, momentum=0.9, eps=1e-5),
            #     nn.ReLU())
        if self.has_softmax:
            self.softmax = nn.Softmax2d()

    def forward(self, x):
        """ Returns softmaxed output in training and ReID vector in inference"""
        """ TODO: Do I need ReLu at the end before softmax?"""
        if self.validation:
            return None  # j: saves processing time in validation by not predicting ReID
        # x = x.contiguous()
        assert self.has_classify  # j: can only train if classification head is initialized
        # x = x.permute([1, 0, 2, 3]).contiguous()
        # batch_size = x.shape[0]
        # x = x.view(batch_size, 128, -1).permute([0, 2, 1]).contiguous()
        # output = self.classify(x)  # j: classify to n_obj_ids classes
        # grid_size = int(math.sqrt(output.shape[1]))
        # output = output.permute(0, 2, 1).contiguous().view(batch_size, self.n_obj_ids, grid_size, grid_size)
        # j: TODO: this version of model only returns embeddings and leaves the Loss function to do classification
        return x

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)  # j: load text file to list of module defs
        self.hyperparams, self.module_list = create_modules(self.module_defs)  # j: create modules from defs
        self.yolo_layers = [layer[0] for layer in self.module_list if isinstance(layer[0], YOLOLayer)]
        self.reid_layers = [layer[0] for layer in self.module_list if isinstance(layer[0], ReIDLayer)]

        self.s_det = nn.Parameter(-1.85 * torch.ones(1))  #j: self-balancing loss from FairMOT
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))  #j: self-balancing loss from FairMOT

        self.seen = 0  # j: number of images already seen during training
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)  # j: (_, _, _, self.seen, _)
        self.validation = False

    def forward(self, x):
        img_size = x.size(2)
        layer_outputs, yolo_outputs, reid_outputs = [], [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            # j: "route" module makes a concatenation of layer_outputs at module_def["layers"] indices
            elif module_def["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            # j: "shortcut" module makes a sum (like residual net) between last layer_outputs layer and output at module_def["from"] index
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                x = module[0](x, img_size)
                yolo_outputs.append(x)
            elif module_def["type"] == "reid":
                x = module(x)
                reid_outputs.append(x)
            layer_outputs.append(x)

        # j: n.b. different returns if in training or evaluation mode
        if reid_outputs != []:
            return (yolo_outputs, reid_outputs) if (self.training or self.validation) else (torch.cat(yolo_outputs, 1), reid_outputs)
        else:
            return (yolo_outputs, None) if (self.training or self.validation) else (torch.cat(yolo_outputs, 1), None)  # j: when no reid module

    def load_darknet_weights(self, weights_path, cutoff=None):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        if cutoff:
            print("Model will load weights up to module {}".format(cutoff))
        else:
            print("Model will load all weights")

        # if "darknet53.conv.74" in weights_path:
        #     cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            # j: get the right module - module[0] (first elt. of sequential)
            # j: use module to get .numel() of module biases
            # j: move data from weights array  to biases at ptr:(ptr+num_b) and advance the pointer
            # j: use module to get .numel() of module weights
            # j: move data from weights array to weights at ptr:(ptr+num_w) and advance the pointer

            if i == cutoff:
                print("Loaded DarkNet weights up until module {}".format(cutoff))
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    # j: use a pointer to look up weights array, which stores all model parameters sequentially
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()

def load_model(model_path, weights_path=None, use_cuda=1, cutoff = None, no_reid_end = False):
    """Loads the yolo model from file.

    :param model_path: Path to model definition file (.cfg)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :return: Returns model
    :rtype: Darknet
    """

    device = torch.device("cuda" if (torch.cuda.is_available() and use_cuda) else "cpu")  # Select device for inference
    model = Darknet(model_path).to(device)
    model.apply(weights_init_normal)
    new_params = model.state_dict()

    # If pretrained weights are specified, start from checkpoint or weight file
    if weights_path:
        if weights_path.endswith(".pth"):  
            # Load checkpoint weights
            loaded_state_dict = torch.load(weights_path)
            if no_reid_end:
                # j: do not load the last reid layer (classifier), when continuing training
                # del loaded_state_dict[f"module_list.{cutoff}.reid_{cutoff}.classify.weight"]  # TODO: debug uncomment
                # del loaded_state_dict[f"module_list.{cutoff}.reid_{cutoff}.classify.bias"]  # TODO: debug uncomment

                # TODO: in "13x13 reduced" model we have these weights
                # module_list.113.reid_113.classify.1.num_batches_tracked
                # module_list.113.reid_113.classify.0.weight
                # module_list.113.reid_113.classify.0.bias

                # Error:
                # module_list.113.reid_113.classify.weight

                del loaded_state_dict[f"module_list.{cutoff}.reid_{cutoff}.classify.0.weight"]  # TODO: debug uncomment
                del loaded_state_dict[f"module_list.{cutoff}.reid_{cutoff}.classify.0.bias"]
                del loaded_state_dict[f"module_list.{cutoff}.reid_{cutoff}.classify.1.num_batches_tracked"]
                del loaded_state_dict[f"module_list.{cutoff}.reid_{cutoff}.classify.1.weight"]
                del loaded_state_dict[f"module_list.{cutoff}.reid_{cutoff}.classify.1.bias"]
                del loaded_state_dict[f"module_list.{cutoff}.reid_{cutoff}.classify.1.running_mean"]
                del loaded_state_dict[f"module_list.{cutoff}.reid_{cutoff}.classify.1.running_var"]
            new_params.update(loaded_state_dict)
            model.load_state_dict(new_params)
        else:  
            # Load darknet weights
            model.load_darknet_weights(weights_path, cutoff)
    return model
