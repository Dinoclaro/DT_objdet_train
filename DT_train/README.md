# **DT_train*

## About the Directory

This directory is used for generating and formatting a dataset specific to Duckietown. The dataset is then used to train a YOLOv5 model. The dataset is made up of real and simulated images from the Duckietown dataset. Real images Simulated images are collected within the `duckietown-lx` environment as it eliminates the need to create Python virtual environments necessary to run `duckietown-gym`. The model is trained using Google Colab to make use of the TPU resources.

## Prerequisites

The list below states the prerequisites to use this directory.

1. Laptop Setup:
   - Ubuntu 22.04 (Recommended)
   - Docker
   - Duckietown shell

2. Assembled Duckiebot: The Duckiebot should be able to boot up. Follow these setup instructions:
   - [Assembly](https://docs.duckietown.com/daffy/opmanual-duckiebot/assembly/db21m/index.html)
   - [Flashing SD Card](https://docs.duckietown.com/daffy/opmanual-duckiebot/setup/setup_sd_card/index.html)
   - [First Boot](https://docs.duckietown.com/daffy/opmanual-duckiebot/setup/setup_boot/index.html)

3. Cloned `duckietown-lx` Repository: The repository can be found [here](https://github.com/duckietown/duckietown-lx).

## Instructions

1. Replace the `data_collection.py` and `setup_activity.py` files in the `duckietown-lx/object-detection/packages/utils` directory with the two files with the same names found in this directory. These files contain modifications that extend the dataset to include labels for duckiebots. The main difference between the `data_collection.py`files is that the modified version in this directory file saves segmented images to the assets directory. The `setup_activity.py` file has modifications that are necessary to add duckiebot labels.

2. Follow the instructions in the `duckietown-lx/object-detection-lx\setup.ipynb` notebook to download the real dataset. Then run the `data_collection.py` using the following two lines below, ensure that your terminal is cd'ed into the `duckietown-lx/object-detection-lx` directory.

    ```shell
    dts code build
       dts code workbench --simulation --launcher data-collection
    ```
The simulation will automatically close when the specified number observations have been obtained. 

3. Copy the images in asset directory of the `duckietown-lx\object-detection` directory across to this directory. 

4. cd into this directory and run the `sim_image.py` and `real_image.py` files. Note that you will have manually add teh path to your directory in these files. This should move all images and labels directory to the `train` and `validation` 

5. Run the `dataset_zip.py`. This should zip up the `duckietown_object_detection_dataset`.

6. Upload dataset to a  Google Drive

Please be aware that
* Do **not** rename the dataset zip file
* The file should be uploaded to the out-most ***"My Drive"*** area

6. Use [this Google Colab Notebook] to train the YOLOv5 model. The notebook walks you through the procedure. 