""" Add unique ids to COCO labels
    Before: cls x1 y1 x2 y2 (in whichever bbox format)
    After cls bbox_id x1 y1 x2 y2 (in whichever bbox format)
    """
import os
import numpy as np

if __name__ == "__main__":
    path = "/home/julius-think/Thesis/Code/YOLO-lindernoren/data/coco/labels/"
    # path = "/home/julius-think/Thesis/Code/YOLO-lindernoren/data/coco/labels/val2014"
    id = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            label_path = os.path.join(root, file)
            _labels = np.loadtxt(label_path).reshape(-1, 5)
            labels = np.empty([len(_labels), 6])
            for idx, label in enumerate(_labels):
                labels[idx] = np.array([label[0], id, *label[1:]])
                id += 1
            np.savetxt(label_path, labels, "%10.6f")