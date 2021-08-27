""" Add unique ids to COCO labels
    Before: cls x1 y1 x2 y2 (in whichever bbox format)
    After cls bbox_id x1 y1 x2 y2 (in whichever bbox format)
    """
import os
import numpy as np

offset0 = 501  # mot15 identities
offset1 = 215  # mot16 identities
# offset2 = 2215  # mot17 identities
offset = offset0 + offset1
if __name__ == "__main__":
    path = "/home/julius-think/Thesis/Code/YOLO-lindernoren/data/mot_combined/mot_extra2.train"
    id = 0
    with open(path, "r") as files:
        for file in files:
            file = file.rstrip().replace("images", "labels_with_ids").replace(".jpg", ".txt")
            try:
                _labels = np.loadtxt(file).reshape(-1, 6)
                _labels[:, 1] = _labels[:, 1] + offset
                new_file = file.replace("mot20", "mot_combined")
                dir = os.path.dirname(new_file)
                os.makedirs(dir, exist_ok=True)
                print(f"Creating: {dir} saving {file}")
                np.savetxt(new_file, _labels, "%10.6f")
            except:
                print(f"{file} not found (proably)")
print(f"{path} labels incremented by {offset} and saved to mot_combined)")