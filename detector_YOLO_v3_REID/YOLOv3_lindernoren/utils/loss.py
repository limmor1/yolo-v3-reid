import math

import torch
import torch.nn as nn

import numpy as np

from .utils import to_cpu
import torch.nn.functional as F
import cv2

# This new loss function is based on https://github.com/ultralytics/yolov3/blob/master/utils/loss.py
def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

img_id = 0  # for debugging purposes
def compute_loss(predictions, reid_predictions, targets, model, n_obj, s_det=1, s_id=1, debug_img=None):  # predictions, targets, model
    # j: calculate three losses and a total - lbox, lobj, lcls and loss (total)
    device = targets.device
    lcls, lbox, lobj, lreid = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
    tcls, tbox, indices, anchors, reid_classes = build_targets(predictions, targets, model)  # targets
    hyperparams = model.hyperparams  # hyperparameters

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))

    # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = smooth_BCE(eps=0.0)  # cp - positive class label factor, cn - negative class label factor

    # Focal loss
    gamma = 0  # hyperparams['fl_gamma']  # focal loss gamma
    if gamma > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, gamma), FocalLoss(BCEobj, gamma)

    IDLoss = nn.CrossEntropyLoss(ignore_index=-1)

    # Losses
    # j: TODO: why balance objectness?
    balance = [4.0, 1.0, 0.4, 0.1]  # P3-P6  # j: balance of objectness score in the loss function of each layer (n.b. last number not used)
    # layer_mults = [4, 2, 1]  j: # Upscale to 52x52
    for layer_index, layer_predictions in enumerate(predictions):  # j: for every of the 3 YOLO output layers
        # layer_mult = layer_mults[layer_index]  j: # Upscale to 52x52
        b, anchor, grid_j, grid_i = indices[layer_index]  # image, anchor, gridy, gridx
        # print(f"layer_predictions.shape: {layer_predictions.shape}")
        tobj = torch.zeros_like(layer_predictions[..., 0], device=device)  # target obj

        num_targets = b.shape[0]  # number of targets
        if num_targets:
            ps = layer_predictions[b, anchor, grid_j, grid_i]  # prediction subset corresponding to targets

            # Regression
            # j: convert from anchors + predictions --> predicted bboxes
            pxy = ps[:, :2].sigmoid() * 2. - 0.5  # j: TODO why *2 - 0.5? Makes range - [-0.5, 1.5]
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[layer_index]
            pbox = torch.cat((pxy, pwh), 1)  # predicted box

            # j: calculate IoU loss (use CIoU)
            iou = bbox_iou(pbox.T, tbox[layer_index], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
            lbox += (1.0 - iou).mean()  # iou loss

            model.gr = 1

            # Objectness
            # j: at each target box location in tobj(images, anchors, 13, 13) store IoU as objectness score
            tobj[b, anchor, grid_j, grid_i] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

            # Classification
            # j: create class target array t. Init with negative class label (cn) and store cp at matching target set locations for each class
            t = torch.full_like(ps[:, 5:], cn, device=device)  # targets  #j: targets of size n_boxes x n_classes, filled with cn as default (negative class label factor for smoothing)
            t[range(num_targets), tcls[layer_index]] = cp
            lcls += BCEcls(ps[:, 5:6], t[:, 0:1])  # BCE  # j: only look at the first class

            # j: loss ReID
            if reid_predictions != None:  # j: False in validation mode (we do not use main loss to validate reids)
                # j: dependent on the prediction scale (52x52, 26x26, 13x13) mark a number of 52x52 output cells as detections\
                #    each marked cells will regress an embedded vector
                # if layer_index == 0:
                #     # Upscale to 52x52
                #     # j: do not repeat cells
                #     # grid_i = grid_i * 4
                #     # grid_j = grid_j * 4
                #     # j: repeat cells
                #     # grid_i_mult = torch.cat([grid_i+i for i in range(layer_mult)], 0)  # j: 4,2,1 consecutive index series
                #     # grid_j_mult = torch.cat([grid_j+j for j in range(layer_mult)], 0)
                #
                #     # Downscale to 13x13
                #     # grid_i_mult = grid_i
                #     # grid_j_mult = grid_j
                #
                #     # Up/downscale to 26x26
                #     grid_i_mult = grid_i * 2
                #     grid_j_mult = grid_j * 2
                # if layer_index == 1:
                #     # Upscale to 52x52
                #     # j: do not repeat cells
                #     # grid_i = grid_i * 2
                #     # grid_j = grid_j * 2
                #     # j: repeat cells
                #     # grid_i_mult = torch.cat([grid_i + i for i in range(layer_mult)], 0)  # j: 4,2,1 consecutive index series
                #     # grid_j_mult = torch.cat([grid_j + j for j in range(layer_mult)], 0)
                #
                #     # Downscale to 13x13
                #     # grid_i_mult = grid_i // 2
                #     # grid_j_mult = grid_j // 2
                #
                #     # Up/downscale to 26x26
                #     grid_i_mult = grid_i
                #     grid_j_mult = grid_j
                #
                # elif layer_index == 2:
                # if layer_index == 2:
                #     # Upscale to 52x52
                #     # grid_i_mult = grid_i
                #     # grid_j_mult = grid_j
                #
                #     # Downscale to 13x13
                #     # grid_i_mult = grid_i // 4
                #     # grid_j_mult = grid_j // 4
                #
                #     # Up/downscale to 26x26
                #     grid_i_mult = grid_i // 2
                #     grid_j_mult = grid_j // 2


                    # Allscale mode
                    grid_i_mult = grid_i
                    grid_j_mult = grid_j


                    # j: only calculate reid loss for unique targets (target - anchor assignment allows multiple!)
                    # reid_class = reid_classes[layer_index]  # reid class for each bounding box in this layer
                    reid_class = reid_classes[layer_index] - 1
                    targs = list(zip(reid_class.to("cpu").numpy(), b.to("cpu").numpy(), grid_j_mult.to("cpu").numpy(), grid_i_mult.to("cpu").numpy()))
                    unique_targets = unique(targs)
                    reid_class_u, b_u, grid_j_u, grid_i_u = zip(*unique_targets)
                    reid_class_u = torch.tensor(reid_class_u, requires_grad=False).to("cuda")
                    # print(f"Unique target class number: {len(reid_class_u)}")  # TODO: temp debug
                    # print(f"Unique target classes: {reid_class_u}")  # TODO: temp debug
                    # print(f"Unique target cells: {str(list(zip(grid_j_u, grid_i_u)))}")  # TODO: temp debug

                    # j: no unique class
                    # reid_class_u, b_u, grid_j_u, grid_i_u = reid_class, b, grid_j_mult, grid_i_mult

                    # print(f"Target class number: {len(reid_class_u)}")  # TODO: temp debug
                    # print(f"Target classes: {reid_class_u}")  # TODO: temp debug
                    # print(f"Target cells: {str(list(zip(grid_j_u, grid_i_u)))}")  # TODO: temp debug

                    # Calculate classification
                    # x = reid_predictions[0][b_u, :, grid_j_u, grid_i_u].contiguous()  # j: n.b. 1-scale
                    x = reid_predictions[layer_index][b_u, :, grid_j_u, grid_i_u].contiguous()  # j: n.b. all-scale
                    # get predictions
                    # x = x.permute([1, 0, 2, 3]).contiguous()
                    # batch_size = x.shape[0]
                    # x = x.view(batch_size, 128, -1).permute([0, 2, 1]).contiguous()

                    # preid = model.reid_layers[0].classify(x)  # j: classify to n_obj_ids classes
                    preid = model.reid_layers[layer_index].classify(x).contiguous()  # j: classify to n_obj_ids classes

                    # reid_class_u = torch.tensor(reid_class_u, requires_grad=False).to("cuda")

                    # # j: DEBUG: mark the targets cell on image that is used for REID
                    # scales = [[13,13], [26,26], [52,52]]
                    # for cell_j, cell_i in zip(grid_j_u, grid_i_u):
                    #     x = int(cell_i * 416 / scales[layer_index][0])
                    #     y = int(cell_j * 416 / scales[layer_index][1])
                    #     x2 = int(x + 416 / scales[layer_index][0])
                    #     y2 = int(y + 416 / scales[layer_index][1])
                    #
                    #     debug_img = np.array(debug_img.astype(np.uint8)).copy()
                    #     cv2.rectangle(debug_img, (x, y), (x2, y2), (255, 0, 0))
                    #     global img_id

                    # My Loss
                    # reid_label = torch.zeros([num_targets, n_obj], dtype=torch.long).to("cuda")
                    # reid_label[[i for i in range(num_targets)], reid_class] = 1  # j: 1-hot encoding
                    # reid_label = torch.cat([reid_label for i in range(layer_mult)], 0)
                    # b = torch.cat([b for i in range(layer_mult)], 0)  # j: upscale to 52x52

                    # preid = preid.log()
                    # reid_layer_loss = (preid * reid_label).sum() / n_obj / -num_targets

                    # FairMOT loss
                    emb_scale = math.sqrt(2) * math.log(n_obj - 1)
                    preid = emb_scale * F.normalize(preid)
                    reid_layer_loss = IDLoss(preid, reid_class_u)
                    lreid += (reid_layer_loss / 3) # j: average over 3 scales
                    # lreid += reid_layer_loss # j: average over 3 scales
            else:
                raise Exception("This should not happen (too often)")
                lreid = torch.tensor([0]).to("cuda")  # j: dummy value for reid loss

        # j: layer_predictions[..., 4] - take the 4th elt of the last dimension (85), keep the rest intact
        # j: ==> takes the objectness prediction, compares it to objectness target, scaled by balance for that ouput layer
        lobj += BCEobj(layer_predictions[..., 4], tobj) * balance[layer_index]  # obj loss

    # j: TODO: DEBUG
    # img_id += 1
    # cv2.imwrite(f"/home/julius-think/Desktop/temp/input_reid_cells_{layer_index}_{img_id}.jpg", debug_img)

    # j: TODO: why these scalings?
    # lbox *= 0.05 * (3. / 2)
    # lobj *= (3. / 2)
    # lcls *= 0.31

    lbox *= 0.048  # j: 1:20 lbox vs lobj loss
    lobj *= 0.952  # j: 1:20 lbox vs lobj loss

    batch_size = reid_predictions[0].shape[0]

    # loss = (lbox + lobj + lcls) + (0.5 * lreid)  # j: weighing ReID loss like in the YOLO-ReID paper
    l_det = (lbox + lobj + lcls) * batch_size
    # l_det = (lbox + lobj + 0)  # j: no class loss
    loss = torch.exp(-s_det) * l_det + torch.exp(-s_id) * lreid + (s_det + s_id)
    loss *= 0.5

    return loss, to_cpu(torch.cat((lbox, lobj, lcls, loss, lreid)))

def unique(sequence):
    """j: source https://stackoverflow.com/questions/9792664/converting-a-list-to-a-set-changes-element-order"""
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

def build_targets(p, targets, model):
    ''' # j:
        p - predictions
        targets - GT boxes for a minibatch of images, size(N, 6) = N x (image,class,x,y,w,h)

    '''
    # Build targets for compute_loss(), input targets (image,class,x,y,w,h)

    # j: init empty vars to gather
    na, nt = 3, targets.shape[0]  # number of anchors, targets
    tcls, tbox, indices, anch, reid_classes = [], [], [], [], []  # j: target classes, boxes, indices, anchors, reid_classes

    # gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
    gain = torch.ones(8, device=targets.device)  # j: move indexes to accommodate unique ID

    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

    g = 0.5  # bias
    off = torch.tensor([[0, 0]], device=targets.device).float() * g  # offsets

    for i, yolo_layer in enumerate(model.yolo_layers):
        # j: go through every YOLO layer and
        anchors = yolo_layer.anchors / yolo_layer.stride  # retrieve the anchors  # j: scale anchors by stride to be in the cell scale
        # gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain  # j: init tensor to 1D with values (4, 3, 13, 13, 85), take x_cells, y_cells to make grid sizes into tensor (_, _, _, 13, 13, 13, 13)
        gain[3:7] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # j: move indexes to accommodate unique ID

        # Match targets to anchors
        t = targets * gain
        if nt:
            # Matches
            # r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
            r = t[:, :, 5:7] / anchors[:, None]  # j: move indexes to accommodate unique ID
            j = torch.max(r, 1. / r).max(2)[0] < 4  # compare #TODO
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            t = t[j]  # filter  j: filter out these ground truths that are more than 4x bigger or smaller than anchor width or height

            # j: unnecessary code after this point
            # Offsets
            # gxy = t[:, 2:4]  # grid xy
            gxy = t[:, 3:5]  # j: move indexes to accommodate unique ID
            # gxi = gain[[2, 3]] - gxy  # inverse
            gxi = gain[[3, 4]] - gxy  # inverse
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T  # j: this is not used anywhere?
            j = torch.stack((torch.ones_like(j),))  # j: this overwrites the above filter...
            t = t.repeat((off.shape[0], 1, 1))[j]   # j: this just takes every value of t...
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]  # j: this does nothing again...
        else:
            t = targets[0]
            offsets = 0

        # Define
        b, c = t[:, :2].long().T  # image, class
        reid_class = t[:, 2].long().T  # j: unique ID of each bounding box (for ReID)
        reid_classes.append(reid_class)
        # gxy = t[:, 2:4]  # grid xy
        gxy = t[:, 3:5]  # j: move indexes to accommodate unique ID
        # gwh = t[:, 4:6]  # grid wh
        gwh = t[:, 5:7]  # j: move indexes to accommodate unique ID grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        # Append
        # a = t[:, 6].long()  # anchor indices
        a = t[:, 7].long()  # j: move indexes to accommodate unique ID
        # indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
        indices.append((b, a, gj.clamp_(0, gain[4] - 1), gi.clamp_(0, gain[3] - 1)))  # j: move indexes to accommodate unique ID
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class

    return tcls, tbox, indices, anch, reid_classes

    # j: data format - all lists of tensors or tuples
    # tcls - [tensor(29), tensor(38), tensor(24)],
    # tbox - [tensor(29,4), tensor(38,4), tensor(24,4)]
    # indices -
    # [
    #   [tensor(29),tensor(29),tensor(29),tensor(29)]
    #   [tensor(29),tensor(29),tensor(29),tensor(29)]
    #   [tensor(29),tensor(29),tensor(29),tensor(29)]
    #   [tensor(29),tensor(29),tensor(29),tensor(29)]

    #   [tensor(38),tensor(38),tensor(38),tensor(38)]
    #   [tensor(38),tensor(38),tensor(38),tensor(38)]
    #   [tensor(38),tensor(38),tensor(38),tensor(38)]
    #   [tensor(38),tensor(38),tensor(38),tensor(38)]

    #   [tensor(24),tensor(24),tensor(24),tensor(24)]
    #   [tensor(24),tensor(24),tensor(24),tensor(24)]
    #   [tensor(24),tensor(24),tensor(24),tensor(24)]
    #   [tensor(24),tensor(24),tensor(24),tensor(24)]
    # ]
    # anch - [tensor(29,2), tensor(38,2), tensor(24,2)


