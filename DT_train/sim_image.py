"""
This script is used to move the simulator images to the training and validation folders.

Disclaimer: Script is modified from Duckietown-lx/object-detection repo.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm


from utils.misc import  train_test_split

# ---------
# Constants
# ---------
# Percentage of simulated data that will go into the training set 
SIMULATED_TRAIN_SPLIT_PERCENTAGE = 0.8

# Dimension of image size for YOLOv5
IMAGE_SIZE = 416

# Directory of dataset
DATASET_DIR="/home/dino/DT_sim/duckietown_object_detection_dataset"

# Constants for resizing images 
npz_index = 0
all_images_names = []

# USE tqdm to create a progress bar 
filenames = tqdm(os.listdir(f"{DATASET_DIR}/labels"))

# Create a list of all images with a valid bounding box 
for filename in filenames:
    other_bboxes = []
    largest_bbox = None
    largest_area = 0

    with open(f"{DATASET_DIR}/labels/{filename}", 'r') as file:
        lines = file.readlines()

    # Iterate through the lines of the file
    for line in lines:
        # Split each line into parts
        parts = line.strip().split()

        # Check if the line corresponds to an object with id 4
        if len(parts) >= 5 and parts[0] == '4':
            # Parse the coordinates and dimensions of the bounding box
            x, y, w, h = map(float, parts[1:5])

            # Calculate the area of the bounding box
            area = w * h

            # Check if this bounding box has a larger area than the current largest
            if area > largest_area:
                largest_area = area
                largest_bbox = (x, y, w, h)
        else:
            # For objects with ids other than 4, store them in the 'other_bboxes' list
            other_bboxes.append(line)

    # Write the largest bounding box and other bounding boxes to a new text file
    with open(f"{DATASET_DIR}/labels/{filename}", 'w') as new_file:
        # Write the other bounding boxes
        new_file.writelines(other_bboxes)

        # Write the largest bounding box (if found)
        if largest_bbox:
            new_file.write(f"4 {largest_bbox[0]} {largest_bbox[1]} {largest_bbox[2]} {largest_bbox[3]}\n")
        
    filename_no_ext = filename.rstrip(".txt")
    all_images_names.append(filename_no_ext)


# Move simulator to training and validation folders 

print("NOW GOING TO MOVE IMAGES INTO TRAIN AND VAL")
train_test_split(all_images_names, SIMULATED_TRAIN_SPLIT_PERCENTAGE, DATASET_DIR)
print("DONE!")