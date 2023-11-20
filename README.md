# **DT_project_60**

## About the Directory

This directory is used for generating and formatting a Duckietown-specific dataset which is then used to train a YOLOv5 model. The dataset is made up of real and simulated images from the Duckietown dataset. Both real and simulated images are collected within the [duckietown-lx](https://github.com/duckietown/duckietown-lx/tree/mooc2022) environment as it eliminates the need to create a Python Virtual environment for the [duckietown-gym](https://github.com/duckietown/gym-duckietown) simulation environment. The model is trained using Google Colab to make use of the TPU resources.

## Prerequisites

The list below states the prerequisites to use this directory.

1. Cloned [duckietown-lx](https://github.com/duckietown/duckietown-lx/tree/mooc2022) repository.

## Instructions

1. Replace the `data_collection.py` and `setup_activity.py` files in the `duckietown-lx/object-detection/` directory with the two files with the same names found in the `replace` folder in this directory. These files contain modifications that extend the dataset to include labels for duckiebots. The main difference between the `data_collection.py` files is that the modified version in this directory file saves segmented images to the assets directory. The `setup_activity.py` file has modifications that are necessary to add duckiebot labels.

2. Follow the instructions in the [duckietown-lx/object-detection-lx\setup.ipynb](https://github.com/duckietown/duckietown-lx/blob/mooc2022/object-detection/notebooks/02-Setup-Data-Collection/setup.ipynb) notebook to download the real dataset. Ensure that your terminal is cd'ed into the `duckietown-lx/` and then run the `data_collection.py` using the following two lines below:

    ```shell
    dts code build
    dts code workbench --simulation --launcher data-collection
    ```
    The simulation will automatically close when the specified number observations have been obtained. 

3. Copy the images in the `asset` directory of the `duckietown-lx\object-detection` directory across to this directory. 

4. cd into this directory and run the `sim_image.py` and `real_image.py` files. Note that you will have to manually add the path to your directory in the `DATASET_DIR` variable for each of these files. This should move all images and labels directory to the `train` and `validation` folders. 

5. Run the `dataset_zip.py`. This should zip up the `duckietown_object_detection_dataset`.
Copy the dataset to a  Google Drive

    Please be aware that
    * Do **not** rename the dataset zip file
    * The file should be uploaded to the out-most ***"My Drive"*** area

6. Use the `DT_training.ipynb` notebook in this directory to train the YOLOv5 model. The notebook walks you through the procedure, after training there should be a folder in your Google drive where you can look at the training results. 