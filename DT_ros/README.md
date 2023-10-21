# **DT_ros**

## About the Directory

This directory contains the ROS agent to run the YOLOv5 model and the Braitenberg controller as Duckietown compliant Docker image. This ROS agent is based on the ROS-agent for the [object-detection](https://github.com/duckietown/duckietown-lx/tree/mooc2022/object-detection) learning experience (lx) and runs within the [duckietown-lx](https://github.com/duckietown/duckietown-lx) environment.  

## Prerequisites

The list below states the prerequisites to use this directory.

1. Trained YOLOv5 model. If not, please refer to the `DT_train` directory is a YOLOv5 model is not changed. 

## Instructions

1. Navigate to the `.dt-shell/recipes/duckietown/duckietown-lx-recipes/mooc2022/object-detection` directory. `.dt-shell` is a hidden folder that should be in `Home` directory. The folder contains recipes and other utilities for running the lx that are not included in the `duckietown-lx` directory.  

2. Replace all the contents of `package` directory with the content in this directory.

3. cd into the `duckietown-lx/object-detection` directory and build the agent by running 

```
dts code build
```
4. The agent can be run in simulation using:

```
dts code workbench --sim
```

Alternatively, it can be run on the duckiebot, where the model runs on your local computer:
```
dts code workbench -b <DUCKIEBOT_NAME> --local
```

or run the entire agent on the duckiebot using:

```
dts code workbench -b <DUCKIEBOT_NAME>
```

