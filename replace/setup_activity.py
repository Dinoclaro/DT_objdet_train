"""
This script contains functions used to extract bounding boxes of the segmented images.

Disclaimer: Script is modified from Duckietown-lx/object-detection repo.
"""

import numpy as np
import cv2

# Define a dictionary with objects and corresponding hexadecimal colour code 
mapping = {
    "house": "3deb34",
    "bus": "aa0000",
    "truck": "961fad",
    "duckie": "cfa923",
    "cone": "ffa600",
    "floor": "000000",
    "grass": "000000",
    "barrier": "000099",
    "duckiebot": "ad0000"
}

# Modify the hexadecimal colour code to list representing RGB colours
    # Example: "house": " 3deb34" --> "house": [61, 235, 52]
mapping = {key: [int(h[i:i + 2], 16) for i in (0, 2, 4)] for key, h in mapping.items()}


def segmented_image_one_class(segmented_img, class_name):
    '''
    Returns a mask of the image with only one class

    Parameters
    ----------
    segmented_img : np.array
        The segmented image
    class_name : str
        The class name
    
    Returns
    -------
    mask : np.array
    '''

    mask = np.all(segmented_img == mapping[class_name], axis=-1)
    #plt.imshow(mask)
    return mask


def find_all_bboxes(mask):
    '''
    Returns a list of bounding box given a mask containing one class

    Parameters
    ----------
    mask : np.array
        The mask of the image with only one class
    
    Returns
    -------
    boxes : list
    '''
    # Convert the image to uint8 grayscale
    gray = mask.astype("uint8")
    gray[mask == True] = 255
    gray[mask == False] = 0

    # Find the contours and hierarchy of nested contours of the image
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    boxes = []

    for index, cnt in enumerate(contours):
        # If the contour is not a top-level contour, continue
        if hierarchy[0, index, 3] != -1:
            continue
        # Get the bounding box of the contour
            # (x, y) is the top-left coordinate of the rectangle
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append([x, y, w + x, h + y])
    boxes = np.array(boxes)
    return boxes


def find_all_boxes_and_classes(segmented_img):
    '''
    Returns a list of bounding boxes and classes given a segmented image

    Parameters
    ----------
    segmented_img : np.array
        The segmented image
    
    Returns
    -------
    all_boxes : list
    all_classes : list
    '''
    classes = ["duckie", "cone", "truck", "bus", "duckiebot"]
    all_boxes = []
    all_classes = []

    # Find the bounding boxes and classes for each class
    for i, class_name in enumerate(classes):
        # Get the mask of the image with only one class
        mask = segmented_image_one_class(segmented_img, class_name)
        boxes = find_all_bboxes(mask)
        all_boxes.extend(list(boxes))

        # Create a list of classes for each bounding box
        classes = np.array([i] * boxes.shape[0])
        all_classes.extend(list(classes))
    return all_boxes, all_classes