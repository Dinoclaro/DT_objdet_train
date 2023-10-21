"""
This script contains functions used for obtaining images for report about the dataset generation process 
""" 
import json
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

from setup_activity import segmented_image_one_class, find_all_bboxes, find_all_boxes_and_classes

# Constants
DATASET_DIR="/home/dino/DT_sim/assets/demo"
IMAGE_SIZE = 416

# Load images for different functions
obs = np.asarray(Image.open(f'{DATASET_DIR}/duckie_not_seg.png'))
obs_seg = np.asarray(Image.open(f'{DATASET_DIR}/13.png'))
obs_seg = cv2.cvtColor(obs_seg, cv2.COLOR_BGR2RGB)
real = np.asarray(Image.open('/home/dino/DT_sim/frame_000280.png'))
duckie_masked_image = segmented_image_one_class(np.asarray(obs_seg),"duckiebot")
img = np.asarray(Image.open("/home/dino/DT_sim/assets/demo/494_not.png"))

def show_image_with_boxes(img, boxes):
    '''
    Given an image and top-left and bottom-right corners of rectangles, adds the bounding box to the image
    '''
    import matplotlib.patches as patches
    fig, ax = plt.subplots()
    ax.imshow(img)
    for box in boxes:
        rect = patches.Rectangle((box[0], box[2]), box[1]-box[0], box[3]-box[2], linewidth=1, edgecolor='w', facecolor='none')
        ax.add_patch(rect)
    plt.show()

def find_all_bboxes_and_show(mask):
    '''
    Finds and displays the contours of an image and returns a list of bounding boxes.
    '''
    # Convert the image to grayscale
    # Convert the image to uint8 grayscale
    gray = mask.astype("uint8")
    gray[mask == True] = 255
    gray[mask == False] = 0

    # Find the contours and hierarchy of nested contours of the image
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    # Create a copy of the image to draw contours on
    image_with_contours = np.zeros((416, 416, 3), dtype=np.uint8)
    image_with_contours = image_with_contours.astype("uint8")
    for index, cnt in enumerate(contours):
        # If the contour is not a top-level contour, continue
        if hierarchy[0, index, 3] != -1:
            continue
        # Draw the contour on the image
        cv2.drawContours(image_with_contours, [cnt], -1, (0, 255, 0), 2)  # Green color, line thickness = 2

    # Display the image with contours
    plt.imshow(image_with_contours)
    plt.axis('off')
    plt.show()

def resize(img):
    copy = img.copy()
    copy = cv2.resize(img, (416, 416))
    plt.imshow(copy)
    plt.show()

def add_bbox(img,filename):
    """
    Adds bounding box to an image given absolute sizes from the Duckietown Dataset 
    """
    with open('/home/dino/DT_sim/duckietown_object_detection_dataset/annotation/final_anns.json') as anns:
        annotations = json.load(anns)

    # Get the original shape of the image
    orig_y, orig_x = img.shape[0], img.shape[1]

    # Create a scale constant based on the image size
    scale_y, scale_x = IMAGE_SIZE/orig_y, IMAGE_SIZE/orig_x

    for detection in annotations[filename]:
        box = detection["bbox"]
        label = detection["cat_name"]

        if label not in ["duckie", "cone", "duckiebot"]:
            continue # Only resize duckie and duckiebot bounding boxes

        orig_x_min, orig_y_min, orig_w, orig_h = box
        x_min = int(orig_x_min)
        y_min = int(orig_y_min)
        x_max = int(x_min + orig_w)
        y_max = int(y_min + orig_h)

        # Draw bounding box
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    plt.imshow(img)
    plt.show()

def remove_bbox():
    """
    This function helps with the case where there are multiple parent contours for a single duckiebot.
    """
    other_bboxes = []
    largest_bbox = None
    largest_area = 0

    with open("/home/dino/DT_sim/assets/demo/494.txt", 'r') as file:
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
    with open("/home/dino/DT_sim/assets/demo/494.txt", 'w') as new_file:
        # Write the other bounding boxes
        new_file.writelines(other_bboxes)

        # Write the largest bounding box (if found)
        if largest_bbox:
            new_file.write(f"4 {largest_bbox[0]} {largest_bbox[1]} {largest_bbox[2]} {largest_bbox[3]}\n")

def read_bounding_boxes_from_file(file_path):
    bounding_boxes = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split each line into space-separated values
            values = line.strip().split()
            if len(values) >= 5:
                x_center, y_center, width, height = map(float, values[1:5])
                bounding_boxes.append([x_center, y_center, width, height])
    return bounding_boxes

def add_bounding_boxes_to_image(image, bounding_boxes):
    """
    Adds bounding box given the yolo format to an image
    """
    # Create a copy of the image to draw bounding boxes on
    image_with_boxes = image.copy()

    for box in bounding_boxes:
        x_center, y_center, width, height = box

        # Convert relative coordinates to absolute coordinates
        image_height, image_width, _ = image.shape
        x = int(x_center * image_width)
        y = int(y_center * image_height)
        w = int(width * image_width)
        h = int(height * image_height)

        # Calculate bounding box coordinates
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        # Draw the bounding box on the image
        color = (0, 255, 0)  # Green color for the bounding boxes
        thickness = 2
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, thickness)

    # Display the image with bounding boxes using Matplotlib
    plt.imshow(image_with_boxes)
    plt.axis('off')
    plt.show()


# Function calls 

# Find bounding boxes for segmented image 
# boxes = find_all_bboxes(duckie_masked_image)

# Show segmented image with bounding boxes
# find_all_bboxes_and_show(duckie_masked_image)

# Add bounding box to a real image from json 
# add_bbox(real,"frame_000280.png")

# Add bounding box to a real image from txt
# boxes = read_bounding_boxes_from_file("/home/dino/DT_sim/assets/demo/494.txt")
# add_bounding_boxes_to_image(img,boxes)