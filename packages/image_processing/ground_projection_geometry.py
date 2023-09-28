"""
Disclaimer: parts of this script are copied from duckietown/dt-core/packages/complete_image_pipeline/include/comp/image_processing/ground_projection_geometry.py
"""
from typing import NewType, Optional, Tuple

import cv2
import numpy as np


class Point:
    """
    Point class. Convenience class for storing ROS-independent 3D points.
    """

    x: float
    y: float
    z: Optional[float]

    def __init__(self, x=None, y=None, z=None):
        self.x = x  #: x-coordinate
        self.y = y  #: y-coordinate
        self.z = z  #: z-coordinate

    def __repr__(self):
        return f"P({self.x}, {self.y}, {self.z})"

    @staticmethod
    def from_message(msg) -> "Point":
        """
        Generates a class instance from a ROS message. Expects that the message has attributes ``x`` and
        ``y``.
        If the message additionally has a ``z`` attribute, it will take it as well. Otherwise ``z`` will be
        set to 0.

        Args:
            msg: A ROS message or another object with ``x`` and ``y`` attributes

        Returns:
            :py:class:`Point` : A Point object

        """
        x = msg.x
        y = msg.y
        try:
            z = msg.z
        except AttributeError:
            z = 0
        return Point(x, y, z)


ImageSpaceResdepPoint = NewType("ImageSpaceResdepPoint", Point)
ImageSpaceNormalizedPoint = NewType("ImageSpaceNormalizedPoint", Point)
GroundPoint = NewType("GroundPoint", Point)

class GroundProjectionTools:
    """
    Projects an image from the duckiebot camera to the ground plane
    Note:
        All pixel and image operations in this class assume that the pixels and images are already
        RECTIFIED. This means that the pixels are already undistorted and the image is already
        cropped to the region of interest. This is done in the duckietown_utils package.

    Args:
        im_width (``int``): Width of the rectified image
        im_height (``int``): Height of the rectified image
        homography (``np.ndarray``): The 3x3 Homography matrix
    
    """

    img_width: int 
    img_heigth: int 
    H: np.ndarray
    Hinv: np.ndarray

    def __init__(self, img_width: int, img_height: int, homography: np.ndarray):
        self.img_width = img_width
        self.img_height = img_height
        H = np.array(homography)
        # Error checking
        if H.shape != (3,3):
            H_rect = H 
            H = H_rect.reshape((3,3))
            #dtu.logger.warning(f"reshaping your homography matrix:\nfrom\n{H_rect}\nto\n{H}")

        self.H = H 
        self.Hinv = np.linalg.inv(homography)

        #dtu.logger.info(f"Initialized GroundProjectionTools with H:\n{self.H}")

    def get_shape(self) -> Tuple[int, int]:
        """
        Returns the shape of the rectified image
        """
        return self.img_width, self.img_height
    
    def vector2pixel(self, vec: ImageSpaceNormalizedPoint) -> ImageSpaceResdepPoint:
        """
        Converts a [0,1]x[0,1] representation to [0,width]x[0, height] 
        (from normalized to image coordinates)

        Args:
            vec (:py:class:'point'): A :py:class:'point' in normalized coordinates (Only x and y)

        Returns: 
            :py:class:'point': A :py:class:'point' in image coordinates (x and y)

        """
        x = self.img_width * vec.x
        y = self.img_height * vec.y
        return ImageSpaceResdepPoint(Point(x, y))
    
    def pixel2vector(self, pixel: ImageSpaceResdepPoint) -> ImageSpaceNormalizedPoint:
        """
        Converts a [0,width]x[0, height] representation to [0,1]x[0,1] 
        (from image to normalized coordinates)

        Args:
            pixel (:py:class:'point'): A :py:class:'point' in image coordinates (Only x and y)

        Returns: 
            :py:class:'point': A :py:class:'point' in normalized coordinates (x and y)

        """
        x = pixel.x / self.img_width
        y = pixel.y / self.img_height
        return ImageSpaceNormalizedPoint(Point(x, y))
    
    def pixel2ground(self, pixel: ImageSpaceNormalizedPoint) -> GroundPoint:
        """
        Converts a normalised pixel to a ground point in the ground plane using homography matrix

        Args:
            pixel (:py:class:'point'): A :py:class:'point' in normalized coordinates (Only x and y)

        Returns:
            :py:class:'point': A :py:class:'point' in ground coordinates (x and y)        
        """

        # Create a 3x1 matrix with the pixel coordinates
        pixel_vec = np.array([pixel.x, pixel.y, 1.0])
        ground_point = np.dot(self.H, pixel_vec)
        x = ground_point[0] 
        y = ground_point[1] 
        z = ground_point[2]
        a = x / z
        b = y / z

        return GroundPoint(Point(a, b, 0.0))
    
    def ground2pixel(self, point: GroundPoint) -> ImageSpaceNormalizedPoint:
        """
        Projects a point on the ground plane to a normalized pixel (``[0, 1] X [0, 1]``) using the
        homography matrix.

        Args:
            point (:py:class:`Point`): A :py:class:`Point` object on the ground plane. Only the ``x`` and
            ``y`` values are used.

        Returns:
            :py:class:`Point` : A :py:class:`Point` object in normalized coordinates. Only the ``x`` and
            ``y`` values are used.

        Raises:
            ValueError: If the input point's ``z`` attribute is non-zero. The point must be on the ground (
            ``z=0``).

        """
        if point.z != 0:
            msg = "This method assumes that the point is a ground point (z=0). "
            msg += f"However, the point is ({point.x},{point.y},{point.z})"
            raise ValueError(msg)

        ground_point = np.array([point.x, point.y, 1.0])
        image_point = np.dot(self.Hinv, ground_point)
        image_point = image_point / image_point[2]

        x = image_point[0]
        y = image_point[1]

        return ImageSpaceNormalizedPoint(Point(x, y))
    
    def bbox2ground(self, bbox: np.ndarray) -> np.ndarray:
        """
        Converts a bounding box to a ground bounding box

        Args:
            bbox (``np.ndarray``): A bounding box in the format ``[x_center, y_center, width, height]``

        Returns:
            ``np.ndarray`` : A bounding box in the format ``[x_center, y_center, width, height]``

        """
        ground_bboxes = []
        
        for bbox in bbox:
            # Extract bbox parameters
            x_center_norm, y_center_norm, width_norm, height_norm = bbox

            # Find corners
            top_left_norm = (float(x_center_norm - width_norm/ 2), float(y_center_norm - height_norm / 2))
            top_right_norm = (float(x_center_norm + width_norm/ 2), float(y_center_norm - height_norm / 2))
            bottom_left_norm = (float(x_center_norm - width_norm/ 2), float(y_center_norm + height_norm / 2))
            bottom_right_norm = (float(x_center_norm + width_norm/ 2), float(y_center_norm + height_norm / 2))
            corners = [top_left_norm, top_right_norm, bottom_left_norm, bottom_right_norm]

            #print('before corners', corners)
            # Convert center point to ground
            center_pixel = ImageSpaceNormalizedPoint(Point(x_center_norm, y_center_norm))
            print("before", center_pixel)
            center_ground = self.pixel2ground(center_pixel)
            print("after", center_ground)

            # Convert corners to ground
            for pixel in corners:
                pixel = ImageSpaceNormalizedPoint(Point(pixel[0], pixel[1]))
                ground_point = self.pixel2ground(pixel)
                # Update the perimeter pixel
                corners[corners.index((pixel.x, pixel.y))] = (ground_point.x, ground_point.y)

            #print('after corners', corners)
            
            # Calculate width
            width = corners[1][0] - corners[0][0]
            print('width', width)
            # Calculate height
            height = corners[2][1] - corners[0][1]
            print('height', height)
            
            ground_box = [abs(center_ground.x), abs(center_ground.y), abs(width), abs(height)]
            #print(ground_box)
            ground_bboxes.append(ground_box)

        return ground_bboxes