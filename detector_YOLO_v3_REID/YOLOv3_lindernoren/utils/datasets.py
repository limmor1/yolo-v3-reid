import glob
import random
import os
import sys
import warnings
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from torch.utils.data import Dataset
import torchvision.transforms as transforms

# FairMOT components
import cv2
import math
import time
from detector_YOLO_v3_REID.YOLOv3_lindernoren.utils.utils2 import xyxy2xywh
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms as T


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class ImageFolder(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.transform = transform

    def __getitem__(self, index):

        img_path = self.files[index % len(self.files)]
        img = np.array(
            Image.open(img_path).convert('RGB'),
            dtype=np.uint8)

        # Label Placeholder
        boxes = np.zeros((1, 5))

        # Apply transforms
        if self.transform:
            img, _ = self.transform((img, boxes))

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self,
                 list_path,         # j: path of a txt file with all image paths
                 img_size=416,      # j: input to resize to
                 multiscale=True,
                 transform=None):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = []       # j: list of all labels corresponding to the images
        for path in self.img_files:
            image_dir = os.path.dirname(path)
            # j: COCO
            # label_dir = "labels".join(image_dir.rsplit("images", 1))
            # j: CH & MOT
            label_dir = "labels_with_ids".join(image_dir.rsplit("images", 1))  # j: split path by "images", 1 = maximum 1 split
            assert label_dir != image_dir, \
                 f"Image path must contain a folder named 'images'! \n'{image_dir}'"
            label_file = os.path.join(label_dir, os.path.basename(path))
            label_file = os.path.splitext(label_file)[0] + '.txt'
            self.label_files.append(label_file)

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        # self.max_size = self.img_size + 3 * 32
        self.max_size = self.img_size   # j: reduced max multiscale, less memory impact during training
        self.batch_count = 0
        self.transform = transform

        # FairMOT attributes
        self.transforms = T.Compose([T.ToTensor()])

        self.height = 416
        self.width = 416
    def letterbox(self, img, height=608, width=1088,
                  color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
        shape = img.shape[:2]  # shape = [height, width]
        ratio = min(float(height) / shape[0], float(width) / shape[1])
        new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
        dw = (width - new_shape[0]) / 2  # width padding
        dh = (height - new_shape[1]) / 2  # height padding
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
        return img, ratio, dw, dh

    def random_affine(self, img, targets=None, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                      borderValue=(127.5, 127.5, 127.5)):
        # j: apply random rotation/scale, translation and shearing on image and targets (labels) within given ranges.
        # j: return affine-transformed image, transformed labels and the transformation matrix M
        # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
        # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

        border = 0  # width of added border (optional)
        height = img.shape[0]
        width = img.shape[1]

        # Rotation and Scale
        R = np.eye(3)
        a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
        # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
        s = random.random() * (scale[1] - scale[0]) + scale[0]
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

        # Translation
        T = np.eye(3)
        T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
        T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

        M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
        imw = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                                  borderValue=borderValue)  # BGR order borderValue

        # Return warped points also
        if targets is not None:
            if len(targets) > 0:
                n = targets.shape[0]
                points = targets[:, 2:6].copy()
                area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

                # warp points
                xy = np.ones((n * 4, 3))
                xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
                xy = (xy @ M.T)[:, :2].reshape(n, 8)

                # create new boxes
                x = xy[:, [0, 2, 4, 6]]
                y = xy[:, [1, 3, 5, 7]]
                xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

                # apply angle-based reduction
                radians = a * math.pi / 180
                reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
                x = (xy[:, 2] + xy[:, 0]) / 2
                y = (xy[:, 3] + xy[:, 1]) / 2
                w = (xy[:, 2] - xy[:, 0]) * reduction
                h = (xy[:, 3] - xy[:, 1]) * reduction
                xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

                # reject warped points outside of image
                # np.clip(xy[:, 0], 0, width, out=xy[:, 0])
                # np.clip(xy[:, 2], 0, width, out=xy[:, 2])
                # np.clip(xy[:, 1], 0, height, out=xy[:, 1])
                # np.clip(xy[:, 3], 0, height, out=xy[:, 3])
                w = xy[:, 2] - xy[:, 0]
                h = xy[:, 3] - xy[:, 1]
                area = w * h
                ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
                i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

                targets = targets[i]
                targets[:, 2:6] = xy[i]

            return imw, targets, M
        else:
            return imw

    def get_data(self, img_path, label_path):
        """j: Function imported from FairMOT """
        # j: apply HSV augmentation, random affine augmentation on img and labels + self.transforms
        height = self.height
        width = self.width
        img = cv2.imread(img_path)  # BGR
        if img is None:
            raise ValueError('File corrupt {}'.format(img_path))
        augment_hsv = True

        # j: augment HSV
        if True:  # j: always augment
            # SV augmentation by 50%
            fraction = 0.50
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            S = img_hsv[:, :, 1].astype(np.float32)
            V = img_hsv[:, :, 2].astype(np.float32)

            a = (random.random() * 2 - 1) * fraction + 1  # j: calc multiplier for Saturation (max 50% change)
            S *= a
            if a > 1:  # j: prevent out of range values
                np.clip(S, a_min=0, a_max=255, out=S)

            a = (random.random() * 2 - 1) * fraction + 1  # j: same for Value
            V *= a
            if a > 1:
                np.clip(V, a_min=0, a_max=255, out=V)

            img_hsv[:, :, 1] = S.astype(np.uint8)
            img_hsv[:, :, 2] = V.astype(np.uint8)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

        # j: add a letterbox to convert any format to (1088, 608)
        h, w, _ = img.shape
        img, ratio, padw, padh = self.letterbox(img, height=height, width=width)

        # j: convert labels to match the letterboxed format
        # Load labels
        if os.path.isfile(label_path):
            labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)

            # Normalized xywh to pixel xyxy format
            labels = labels0.copy()
            labels[:, 2] = ratio * w * (labels0[:, 2] - labels0[:, 4] / 2) + padw
            labels[:, 3] = ratio * h * (labels0[:, 3] - labels0[:, 5] / 2) + padh
            labels[:, 4] = ratio * w * (labels0[:, 2] + labels0[:, 4] / 2) + padw
            labels[:, 5] = ratio * h * (labels0[:, 3] + labels0[:, 5] / 2) + padh
        else:
            labels = np.array([])

        # Augment image and labels
        if True:  # j: always augment
            img, labels, M = self.random_affine(img, labels, degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.50, 1.20))

        plotFlag = False
        if plotFlag:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.figure(figsize=(50, 50))
            plt.imshow(img[:, :, ::-1])
            plt.plot(labels[:, [1, 3, 3, 1, 1]].T, labels[:, [2, 2, 4, 4, 2]].T, '.-')
            plt.axis('off')
            plt.savefig('test.jpg')
            time.sleep(10)

        # j: convert labels back to xywh format (and add a random flip)
        nL = len(labels)
        if nL > 0:
            # convert xyxy to xywh
            labels[:, 2:6] = xyxy2xywh(labels[:, 2:6].copy())  # / height
            labels[:, 2] /= width
            labels[:, 3] /= height
            labels[:, 4] /= width
            labels[:, 5] /= height
        if True:  # j: always augment
            # random left-right flip
            lr_flip = True
            if lr_flip & (random.random() > 0.5):
                img = np.fliplr(img)
                if nL > 0:
                    labels[:, 2] = 1 - labels[:, 2]

        img = np.ascontiguousarray(img[:, :, ::-1])  # BGR to RGB

        # j: apply transforms
        if self.transforms is not None:
            img = self.transforms(img)

        return img, labels, img_path, (h, w)


    def __getitem2__(self, index):
        """ j: FairMOT variant of dataset loading """
        try:
            # j: convert relative path to absolute if necessary (for crowdhuman_j)
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            cwd = os.getcwd()
            img_path = os.path.join(cwd, "data/", img_path)
            # img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception as e:
            print(f"Could not read image '{img_path}'.")
            return

        try:
            # j: convert relative path to absolute if necessary (for crowdhuman_j)
            cwd = os.getcwd()
            label_path = self.label_files[index % len(self.img_files)].rstrip()
            label_path = os.path.join(cwd, "data/", label_path)

            # # Ignore warning if file is empty
            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore")
            #     # boxes = np.loadtxt(label_path).reshape(-1, 5)
            #     boxes = np.loadtxt(label_path).reshape(-1, 6)  # j: one extra item for ID
            #     # j: limit the number of boxes in the input to self.max_object (for memory reasons)
            #     if len(boxes) > self.max_objects:
            #         boxes = np.random.permutation(boxes)
            #         boxes = boxes[:self.max_objects]
            #         # print(f"Too many bounding boxes in image, taking only a sample of {self.max_objects}...")
        except Exception as e:
            print(f"Could not read label, generating empty label set '{label_path}'.")
            return img_path, torch.zeros([3, 416, 416]), torch.zeros((0, 7))

        img, bb_targets_, img_path, (h, w) = self.get_data(img_path, label_path)

        bb_targets = np.zeros([bb_targets_.shape[0], 7])
        if bb_targets_.size != 0:
            bb_targets[:, 1:] = bb_targets_
        return img_path, img, torch.from_numpy(bb_targets)


    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        try:
            # j: convert relative path to absolute if necessary (for crowdhuman_j)
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            cwd = os.getcwd()
            img_path = os.path.join(cwd, "data/", img_path)
            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception as e:
            print(f"Could not read image '{img_path}'.")
            return

        # ---------
        #  Label
        # ---------
        try:
            # j: convert relative path to absolute if necessary (for crowdhuman_j)
            cwd = os.getcwd()
            label_path = self.label_files[index % len(self.img_files)].rstrip()
            label_path = os.path.join(cwd, "data/", label_path)

            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # boxes = np.loadtxt(label_path).reshape(-1, 5)
                boxes = np.loadtxt(label_path).reshape(-1, 6)  # j: one extra item for ID
                # j: limit the number of boxes in the input to self.max_object (for memory reasons)
                if len(boxes) > self.max_objects:
                    boxes = np.random.permutation(boxes)
                    boxes = boxes[:self.max_objects]
                    # print(f"Too many bounding boxes in image, taking only a sample of {self.max_objects}...")
        except Exception as e:
            print(f"Could not read label, generating empty label set '{label_path}'.")
            return img_path, img, torch.zeros((0, 7))

        # -----------
        #  Transform
        # -----------
        if self.transform:
            img, bb_targets = self.transform((img, boxes))

            # try:
            #     img, bb_targets = self.transform((img, boxes))
            # except:
            #     print(f"Could not apply transform.")
            #     return
        return img_path, img, bb_targets

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if (data != None) and (data[2].shape[0] != 0)]

        if batch != []:  # j: prevent missing batches TODO: why occurs?
            paths, imgs, bb_targets = list(zip(*batch))

            # Selects new image size every tenth batch
            if self.multiscale and self.batch_count % 10 == 0:
                self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))

            # Resize images to input shape
            imgs = torch.stack([resize(img, self.img_size) for img in imgs])

            # Add sample index to targets
            for i, boxes in enumerate(bb_targets):
                boxes[:, 0] = i
            bb_targets = torch.cat(bb_targets, 0)

            return paths, imgs, bb_targets
        else:
            return None, None, None

    def __len__(self):
        return len(self.img_files)
