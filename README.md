# **DT_project_60**

## About the Repository

This repository contains two directories of code used in Project 60 of the 2023 MEC4128S UCT course. This project implements a YOLOv5 model and Bratienberg controller on the duckiebot. [DT_train](../DT_train/README.md) contains various python files that generate and format the dataset as well as a Google Colab notebook used to train the YOLOv5 model. [DT_ros](../DT_ros/README.md) contains the ROS agent as a Duckietown compliant Docker image. 

## Prerequisites

1. Laptop Setup:
   - Ubuntu 22.04 (Recommended)
   - Docker
   - Duckietown shell

2. Assembled Duckiebot: The Duckiebot should be able to boot up. Follow these setup instructions:
   - [Assembly](https://docs.duckietown.com/daffy/opmanual-duckiebot/assembly/db21m/index.html)
   - [Flashing SD Card](https://docs.duckietown.com/daffy/opmanual-duckiebot/setup/setup_sd_card/index.html)
   - [First Boot](https://docs.duckietown.com/daffy/opmanual-duckiebot/setup/setup_boot/index.html)
   - [Manual Control](https://docs.duckietown.com/daffy/opmanual-duckiebot/operations/make_it_move/index.html)


## Instructions

1. Clone the `duckietown-lx`: the repository can be found [here](https://github.com/duckietown/duckietown-lx) and outlines provides instructions to clone the repository.

2. Navigate to the `DT_train` directory in this repository. The directory outlines the procedure to generate the dataset and then train the YOLOv5 model.