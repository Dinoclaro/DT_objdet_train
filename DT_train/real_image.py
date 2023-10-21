"""
This script is used to resize th real images and move them to the training and validation folders.

Disclaimer: Script is modified from Duckietown-lx/object-detection repo.
"""

import json
import os
import cv2
import numpy as np
from tqdm import tqdm

from utils.misc import xminyminxmaxymax2xywfnormalized, train_test_split

# ---------
# Constants
# ---------
# Percentage of simulated data that will go into the training set 
REAL_TRAIN_TEST_SPLIT_PERCENTAGE = 0.8

# Dimension of image size for YOLOv5
IMAGE_SIZE = 416

# Directory of dataset
DATASET_DIR="/home/dino/DT_sim/duckietown_object_detection_dataset"

# Constants for resizing images 
npz_index = 0
all_images_names = []

# ---------
# Load labels from json
# ---------
with open(f"{DATASET_DIR}/annotation/final_anns.json") as anns:
    annotations = json.load(anns)

# ---------
# Save images
# ---------
def save_img(img, boxes, classes):
    global npz_index
    cv2.imwrite(f"{DATASET_DIR}/images/real_{npz_index}.jpg", img)
    with open(f"{DATASET_DIR}/labels/real_{npz_index}.txt", "w") as f:
        for i in range(len(boxes)):
            f.write(f"{classes[i]} "+" ".join(map(str,boxes[i]))+"\n")
    npz_index += 1
    all_images_names.append(f"real_{npz_index}")

# ---------
# Resize images and bounding boxes
# ---------
# USE tqdm to create a progress bar 
filenames = tqdm(os.listdir(f"{DATASET_DIR}/frames"))

# RESIZE real images 
for filename in filenames:
    # read image
    img = cv2.imread(f"{DATASET_DIR}/frames/{filename}")

    # Get the original shape of the image
    orig_y, orig_x = img.shape[0], img.shape[1]

    # Create a scale constant based on the image size
    scale_y, scale_x = IMAGE_SIZE/orig_y, IMAGE_SIZE/orig_x

    # Resize image 
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    # RESIZE the bounding box
    boxes = []
    classes = []

    # Continue if the image does not have nay detections
    if filename not in annotations: 
        continue 
    
    # Get the bounding box if there is a detection
    for detection in annotations[filename]:
        box = detection["bbox"]
        label = detection["cat_name"]

        if label not in ["duckie", "cone", "duckiebot"]:
            continue # Only resize duckie and duckiebot bounding boxes

        orig_x_min, orig_y_min, orig_w, orig_h = box 

        # Scale bounding box 
        x_min = int(np.round(orig_x_min*scale_x))
        y_min = int(np.round(orig_y_min*scale_y))
        x_max = x_min + int(np.round(orig_w*scale_x))
        y_max = y_min + int(np.round(orig_h*scale_y))

        # Append bounding box
        boxes.append([x_min, y_min, x_max, y_max])
        classes.append(1 if label == "duckie" else (2 if label == "cone" else 5))


    if len(boxes) == 0:
        continue

    # Normalise bounding box for YOLOv5
    boxes = np.array([xminyminxmaxymax2xywfnormalized(box, IMAGE_SIZE) for box in boxes])
    classes = np.array(classes) - 1

    # Save the resized images to the "images" folder
    save_img(img, boxes, classes)


# Move resized images to training and validation folders 
train_test_split(all_images_names, REAL_TRAIN_TEST_SPLIT_PERCENTAGE, DATASET_DIR)
print("DONE!")