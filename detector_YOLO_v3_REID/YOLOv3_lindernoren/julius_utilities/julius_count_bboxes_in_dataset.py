""" Read the image paths file:
        - count total unique identities
        - count per-image unique identities
        - show min ID
        - alert if there are repeating IDs in other directories
        - create empty label file if image has no matching label file
    Mark mismatch between image paths and label paths.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == "__main__":
    # list_path = "/home/julius-think/Thesis/Code/YOLO-lindernoren/data/crowdhuman/crowdhuman.train"
    # list_path = "/home/julius-think/Thesis/Code/YOLO-lindernoren/data/crowdhuman/crowdhuman_reduced_1.train"
    # list_path = "/home/julius-think/Thesis/Code/YOLO-lindernoren/data/mot15/mot15.train"
    list_path = "/home/julius-think/Thesis/Code/YOLO-lindernoren/data/mot_combined/mot_combined.train"
    # list_path = "/home/julius-think/Thesis/Code/YOLO-lindernoren/data/mot16/mot16.val"
    # list_path = "/home/julius-think/Thesis/Code/YOLO-lindernoren/data/mot_combined/mot_combined.train"
    # list_path = "/home/julius-think/Thesis/Code/YOLO-lindernoren/data/mot20/mot20_testing_full.train"
    # list_path = "/home/julius-think/Thesis/Code/YOLO-lindernoren/data/coco/trainvalno5k.txt"
    with open(list_path, "r") as file:
        img_files = file.readlines()

    label_files = []  # j: list of all labels corresponding to the images
    for path in img_files:
        image_dir = os.path.dirname(path)
        label_dir = "labels_with_ids".join(
        # label_dir = "labels".join(
            image_dir.rsplit("images", 1))  # j: split path by "images", 1 = maximum 1 split
        assert label_dir != image_dir, \
            f"Image path must contain a folder named 'images'! \n'{image_dir}'"
        label_file = os.path.join(label_dir, os.path.basename(path))
        label_file = os.path.splitext(label_file)[0] + '.txt'
        label_files.append(label_file)

# j: find out how many unique objects in the dataset
max_id = 0
min_id = math.inf
max_nb = 0
# nb_stats = {}
# for i in range(378):
#     nb_stats[i] = 0
paths = {}
error_ct = 0
error_paths = []
for label_path in label_files:
    cwd = os.getcwd()
    label_path = os.path.join(cwd, "data/", label_path)
    dir = os.path.dirname(label_path)
    file = os.path.basename(label_path)
    paths.setdefault(dir, set())
    if not os.path.exists(label_path):
        error_ct += 1
        error_paths.append(label_path)
        print(f"Label path not found: {label_path}")
        print(f"Generating empty label .txt file...")
        with open(label_path, "w") as newfile:
            newfile.write("\n")
    with open(label_path, "r") as file:
        # boxes = np.loadtxt(label_path).reshape(-1, 5)
        boxes = np.loadtxt(label_path).reshape(-1, 6)
        ids = boxes[:, 1]
        for id in ids:
            for path in paths.keys():
                if path != dir:
                    if id in paths[path]:
                        raise Exception(f"Repeating ID {id} in path {label_path} \n Already exists in {path}!")
            else:
                paths[dir].add(id)
        old_max_id = max_id
        old_min_id = min_id
        old_max_nb = max_nb
        n_boxes = len(boxes)
        if len(boxes) != 0:
            max_id = max(int(max(boxes[:, 1])), max_id)
            min_id = min(int(min(boxes[:, 1])), min_id)
            max_nb = max(n_boxes, max_nb)
            # nb_stats[n_boxes] += 1
            if max_id > old_max_id:
                print(f"New max {max_id}")
            if min_id < old_min_id:
                print(f"New min {min_id}")
    # except:
    #     error_ct += 1
    #     error_paths.append(label_path)
    #     print(f"Label path not found: {label_path}")
    #     # print(f"Generating empty label .txt file...")
    #     # os.system("echo   > {label_path}")

print(f"Maximum bbox ID in the dataset = {max_id}")
print(f"Minimum bbox ID in the dataset = {min_id}")
print(f"Maximum number of boxes in an image = {max_nb}")

print(f"Read errors: {error_ct}")
print(f"Total paths: {len(label_files)}")

# Show histogram counts of no. boxes per frame
# plt.figure()
# for key,val in nb_stats.items():
#     if val > 3:
#         plt.scatter(key, val)
# plt.show()