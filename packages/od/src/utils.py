from typing import Tuple


def NUMBER_FRAMES_SKIPPED() -> int:
    # Integer: Returns the number of frames skipped between each prediction 
    return 5


def filter_by_classes(pred_class: int) -> bool:
    """
    Remember the class IDs:

        | Object    | ID    |
        | ---       | ---   |
        | Duckie    | 0     |
        | Cone      | 1     |
        | Truck     | 2     |
        | Bus       | 3     |


    Args:
        pred_class: the class of a prediction
    Returns:
        True for classes to be included, False otherwise
    """
    # Right now, this returns True for every object's class
    # TODO: Change this to only return True for duckies!
    # In other words, returning False means that this prediction is ignored.
    if (pred_class == 0) | (pred_class == 1):
        return True


def filter_by_scores(score: float) -> bool:
    """
    Args:
        score: the confidence score of a prediction
    Returns:
        True for scores to be included above threshold, False otherwise
    """

    if (score > 0.7):
        return True


def filter_by_bboxes(bbox: Tuple[int, int, int, int]) -> bool:
    """
    Args:
        bbox: is the bounding box of a prediction, in xyxy format
                This means the shape of bbox is (leftmost x pixel, topmost y, rightmost x, bottommost y)
    Returns:
        True for bounding box area to be included, False otherwise
    """
    if abs((bbox[2]-bbox[0])*(bbox[3]-bbox[1])) > 100:
        return True