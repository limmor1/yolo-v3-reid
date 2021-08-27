import torch
import torch.nn.functional as F
import numpy as np

import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from .utils import xywh2xyxy_np
import torchvision.transforms as transforms
global clipped_boxes

class ImgAug(object):
    def __init__(self, augmentations=[]):
        self.augmentations = augmentations

    def __call__(self, data):
        # Unpack data
        img, boxes = data

        # Convert xywh to xyxy
        boxes = np.array(boxes)
        # boxes[:, 1:] = xywh2xyxy_np(boxes[:, 1:])
        boxes[:, 2:] = xywh2xyxy_np(boxes[:, 2:])  # j: move indexes to accommodate unique ID

        # Convert bounding boxes to imgaug
        bounding_boxes = BoundingBoxesOnImage(
            # [BoundingBox(*box[1:], label=box[0]) for box in boxes],
            [BoundingBox(*box[2:], label=box[0]) for box in boxes],  # j: move indexes to accommodate unique ID
            shape=img.shape)

        # Apply augmentations
        img, bounding_boxes = self.augmentations(
            image=img,
            bounding_boxes=bounding_boxes)

        # Clip out of image boxes
        # j: n.b. boxes that are completely out of image will be removed from the list
        unique_ids = boxes[:, 1]  # j: keep unique ids intact
        clipped_mask = np.array([bb.is_out_of_image(img) for bb in bounding_boxes])  # j: test with empty mask (if bounding_boxes is None)
        bounding_boxes = bounding_boxes.clip_out_of_image()

        # Convert bounding boxes back to numpy
        # boxes = np.zeros((len(bounding_boxes), 5))
        boxes = np.zeros((len(bounding_boxes), 6))  # j: move indexes to accommodate unique ID
        try:
            boxes[:, 1] = unique_ids[~clipped_mask]  # j: mask out clipped out boxes
        except:
            print("Impossible to invert clipped_mask. Perhaps there are no bounding_boxes after clipping out-of-image boxes? Ignoring...")

        for box_idx, box in enumerate(bounding_boxes):
            # Extract coordinates for unpadded + unscaled image
            x1 = box.x1
            y1 = box.y1
            x2 = box.x2
            y2 = box.y2

            # Returns (x, y, w, h)
            boxes[box_idx, 0] = box.label
            # boxes[box_idx, 1] = ((x1 + x2) / 2)
            # boxes[box_idx, 2] = ((y1 + y2) / 2)
            # boxes[box_idx, 3] = (x2 - x1)
            # boxes[box_idx, 4] = (y2 - y1)
            # j: move indexes to accommodate unique ID
            boxes[box_idx, 2] = ((x1 + x2) / 2)
            boxes[box_idx, 3] = ((y1 + y2) / 2)
            boxes[box_idx, 4] = (x2 - x1)
            boxes[box_idx, 5] = (y2 - y1)

        return img, boxes


class RelativeLabels(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        w, h, _ = img.shape
        # boxes[:,[1,3]] /= h
        boxes[:, [2,4]] /= h  # j: move indexes to accommodate unique ID
        # boxes[:,[2,4]] /= w
        boxes[:, [3,5]] /= w  # j: move indexes to accommodate unique ID
        return img, boxes


class AbsoluteLabels(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        """ @params:
                data - list[img(h, w, ch), nx7 boxes(img_id, cls, box_id, x1, y1, x2, y2)]
        """
        img, boxes = data
        w, h, _ = img.shape
        # boxes[:,[1,3]] *= h
        boxes[:,[2,4]] *= h  # j: move indexes to accommodate unique ID
        # boxes[:,[2,4]] *= w
        boxes[:,[3,5]] *= w   # j: move indexes to accommodate unique ID
        return img, boxes


class PadSquare(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.PadToAspectRatio(
                1.0,
                position="center-center").to_deterministic()
            ])


class ToTensor(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(img)

        # bb_targets = torch.zeros((len(boxes), 6))
        bb_targets = torch.zeros((len(boxes), 7))  # j: move indexes to accommodate unique ID
        bb_targets[:, 1:] = transforms.ToTensor()(boxes)

        return img, bb_targets


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        img, boxes = data
        img = F.interpolate(img.unsqueeze(0), size=self.size, mode="nearest").squeeze(0)
        return img, boxes


DEFAULT_TRANSFORMS = transforms.Compose([
        AbsoluteLabels(),
        PadSquare(),
        RelativeLabels(),
        ToTensor(),
    ])