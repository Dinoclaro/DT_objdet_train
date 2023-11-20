"""
This script contains functions to format the dataset according to the YOLOv5 training format followed by zipping the file.

Disclaimer: Script is modified from Duckietown-lx/object-detection repo.
"""

import os
import tempfile
import shutil
from typing import List
from datetime import datetime


def zip_sub_dirs(abs_root_dir: str, lst_rel_subdirs: List[str], output_basename: str) -> str:
    """Zip some sub-directories, return the zipped file's path"""

    # check no identical output file exists
    out_full = f"{output_basename}.zip"
    if os.path.exists(out_full):
        print(f"File already exists at: {out_full}")
        print("Rename/Move it to run.\nNo operations performed.")
        return ""

    # make temporary directory
    tmp_dir = tempfile.mkdtemp()
    print(f"[{datetime.now()}] Temporary directory created at: {tmp_dir}")

    # format subdir original and temporary paths
    original_paths = [os.path.join(abs_root_dir, _d) for _d in lst_rel_subdirs]
    tmp_paths = [os.path.join(tmp_dir, _d) for _d in lst_rel_subdirs]

    print(f"[{datetime.now()}] List of directories to include in the zip file:")
    # ensure all specified subdirs exist
    for subdir in original_paths:
        assert os.path.exists(subdir), f"Specified path does not exist: {subdir}\nAbort! No operations performed."
        print(f" - {subdir}")

    print(f"[{datetime.now()}] Move subdirs to the temp root dir")
    # move subdirs to the tmp dir
    for ori, tmp in zip(original_paths, tmp_paths):
        shutil.move(ori, tmp)

    # create the zip archive
    print(f"[{datetime.now()}] Compressing and creating the archive...")
    ret = shutil.make_archive(output_basename, 'zip', tmp_dir)
    
    print(f"[{datetime.now()}] Move subdirs back to original location")
    # move directories back to original location
    for tmp, ori in zip(tmp_paths, original_paths):
        shutil.move(tmp, ori)
    
    print(f"[{datetime.now()}] Finished. Archive created at: {ret}")
    return ret


# zip file basename for our dataset
ZIPPED_DATASET_BASENAME_FILE = "duckietown_object_detection_dataset"
# file/dir location constants
DATASET_DIR = "/Path/to/duckietown_object_detection_dataset"
# path and file name (without file extension)
ZIPPED_DATASET_BASENAME_FULL = os.path.join(DATASET_DIR, ZIPPED_DATASET_BASENAME_FILE)
TRAIN_DIR = "train"
VALIDATION_DIR = "val"

_ = zip_sub_dirs(
    abs_root_dir=DATASET_DIR,
    lst_rel_subdirs=[TRAIN_DIR, VALIDATION_DIR],
    output_basename=ZIPPED_DATASET_BASENAME_FULL,
)