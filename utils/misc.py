"""
This script contains miscellaneous functions used in other scripts.

Disclaimer: Script is modified from Duckietown-lx/object-detection repo.
"""

import contextlib
import os
import subprocess
import matplotlib.pyplot as plt

import numpy as np
import cv2

# ---------
# Directory utilities 
# ---------
def run(input, exception_on_failure=False):
    try:
        return subprocess.check_output(
            f"{input}", shell=True, universal_newlines=True, stderr=subprocess.STDOUT
        )
    except Exception as e:
        if exception_on_failure:
            raise e
        return e.output


def runp(input, exception_on_failure=False):
    print(input)
    output = run(input, exception_on_failure)
    if len(output) > 0:
        print(output)

# Move resized images to training and validation folders 
def train_test_split(filenames, split_percentage, dataset_dir):
    train_txt = np.array(filenames)
    # Shuffle the images
    np.random.shuffle(train_txt)
    nb_things = len(train_txt)
    sp = int(split_percentage * nb_things)
    train_txt, val_txt = train_txt[:sp], train_txt[sp:]

    print("ALL IMAGE NAMES TO MOVE DURING THIS SPLIT:", filenames)
    print("DATASET DIRECTORY", dataset_dir)

    def mv(img_name, to_train):
        print("MOVING IMG NAMED", img_name)

        dest = "train" if to_train else "val"
        runp(f"mv {dataset_dir}/images/{img_name}.jpg {dataset_dir}/{dest}/images/{img_name}.jpg")
        runp(f"mv {dataset_dir}/labels/{img_name}.txt {dataset_dir}/{dest}/labels/{img_name}.txt")

    for img in train_txt:
        mv(img, True)
    for img in val_txt:
        mv(img, False)


@contextlib.contextmanager
def makedirs(name):
    try:
        os.makedirs(name)
    except:
        pass
    yield None


@contextlib.contextmanager
def directory(name):
    ret = os.getcwd()
    os.chdir(name)
    yield None
    os.chdir(ret)


# ---------
#  Image utilities 
# ---------
# make boxes to xywh format for YOLOv5:
def xminyminxmaxymax2xywfnormalized(box, image_size):
    xmin, ymin, xmax, ymax = np.array(box, dtype=np.float64)
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    width = xmax - xmin
    height = ymax - ymin

    normalized = np.array([center_x, center_y, width, height]) / image_size
    return np.round(normalized, 5)

def display_img_seg_mask(real_img, seg_img):
    all = np.concatenate((cv2.cvtColor(real_img, cv2.COLOR_RGB2BGR), seg_img), axis=1)

    cv2.imshow("image", all)
    cv2.waitKey(0)









