import os

if __name__ == "__main__":
    # path = "/data/mot16/images/valid/MOT17-11-SDP"
    path = "/home/julius-think/Thesis/Code/YOLO-lindernoren/data/mot15/images/train"

    paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if (file.endswith(".jpg")):
                path = os.path.join(root, file)
                print(path)
                paths.append(path)
    with open("data/mot15/mot15.train", "w") as f:
        f.write("\n".join(paths))