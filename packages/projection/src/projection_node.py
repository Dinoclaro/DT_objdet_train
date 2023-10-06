#!/usr/bin/env python3

import os
from typing import Optional

import cv2
import numpy as np
import yaml

import rospy
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import Vector2D, Segment, SegmentList
from geometry_msgs.msg import Point as PointMsg
from image_processing.ground_projection_geometry import GroundProjectionGeometry, Point
from image_processing.rectification import Rectify
from std_msgs.msg import Float32MultiArray
from nn_model.constants import IMAGE_SIZE
#from custom_msgs.msg import BoundingBox, BoundingBoxList
from sensor_msgs.msg import CameraInfo, CompressedImage

# TODO: Not subscribing on the duckiebot or simulator 

class ProjectionNode(DTROS):
    """
    This node projects the line segments detected in the image to the ground plane and in the robot's
    reference frame.
    In this way it enables lane localization in the 2D ground plane. This projection is performed using the
    homography
    matrix obtained from the extrinsic calibration procedure.

    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that ROS will use

    Subscribers:
        ~camera_info (:obj:`sensor_msgs.msg.CameraInfo`): Intrinsic properties of the camera. Needed for
        rectifying the segments.
        ~lineseglist_in (:obj:`duckietown_msgs.msg.SegmentList`): Line segments in pixel space from
        unrectified images

    Publishers:
        ~lineseglist_out (:obj:`duckietown_msgs.msg.SegmentList`): Line segments in the ground plane
        relative to the robot origin
        ~debug/ground_projection_image/compressed (:obj:`sensor_msgs.msg.CompressedImage`): Debug image
        that shows the robot relative to the projected segments. Useful to check if the extrinsic
        calibration is accurate.
    """

    bridge: CvBridge
    ground_projector: Optional[GroundProjectionGeometry]
    rectifier: Optional[Rectify]

    def __init__(self, node_name: str):
        # Initialize the DTROS parent class
        super(ProjectionNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)

        self.bridge = CvBridge()
        self.ground_projector = None
        self.rectifier = None
        self.homography = self.load_extrinsics()
        self.log(f"Loaded homography matrix: {np.array(self.homography).reshape((3, 3))}")
        self.first_processing_done = False
        self.veh = rospy.get_namespace().strip("/")
        self.log(f"Vehicle name: {self.veh}")
        self.camera_info_received = False

        # subscribers
        camera_info_topic = f"/{self.veh}/camera_node/camera_info"
        self.sub_camera_info = rospy.Subscriber(camera_info_topic, CameraInfo, self.cb_camera_info, queue_size=1)

        sub_detection_topic = f"/{self.veh}/object_detection_node/detections"
        self.sub_detections_ = rospy.Subscriber(sub_detection_topic, Float32MultiArray, self.detections_cb, queue_size=1
        )

        # publishers
        self.pub_ground_detections = rospy.Publisher(
            "~detections_out", SegmentList, queue_size=1, dt_topic_type=TopicType.PERCEPTION
        )
        self.pub_debug_img = rospy.Publisher(
            "~debug/ground_projection_image/compressed",
            CompressedImage,
            queue_size=1,
            dt_topic_type=TopicType.DEBUG,
        )

        self.bridge = CvBridge()

        self.debug_img_bg = None

    def cb_camera_info(self, msg: CameraInfo):
        """
        Initializes a :py:class:`image_processing.GroundProjectionGeometry` object and a
        :py:class:`image_processing.Rectify` object for image rectification

        Args:
            msg (:obj:`sensor_msgs.msg.CameraInfo`): Intrinsic properties of the camera.

        """
        if not self.camera_info_received:
            self.rectifier = Rectify(msg)
            self.ground_projector = GroundProjectionGeometry(
                im_width=IMAGE_SIZE, im_height=IMAGE_SIZE, homography=np.array(self.homography).reshape((3, 3))
            )
            self.log("CameraInfo received.")
            self.camera_info_received = True
        

    def pixel_msg_to_ground_msg(self, point_msg) -> PointMsg:
        """
        Creates a :py:class:`ground_projection.Point` object from a normalized point message from an
        unrectified
        image. It converts it to pixel coordinates and rectifies it. Then projects it to the ground plane and
        converts it to a ROS Point message.

        Args:
            point_msg (:obj:`geometry_msgs.msg.Point`): Normalized point coordinates from an unrectified
            image.

        Returns:
            :obj:`geometry_msgs.msg.Point`: Point coordinates in the ground reference frame.

        """
        # normalized coordinates to pixel:
        norm_pt = Point.from_message(point_msg)
        # pixel = self.ground_projector.vector2pixel(norm_pt)
        # # rectify
        # rect = self.rectifier.rectify_point(pixel)
        # # convert to Point
        # rect_pt = Point.from_message(rect)
        # rect_pt = self.ground_projector.pixel2vector(rect)
        # self.log(f"-------------------------------------->Rectified point: {rect_pt}")
        # project on ground
        ground_pt = self.ground_projector.pixel2ground(norm_pt)
        # point to message
        ground_pt_msg = PointMsg()
        ground_pt_msg.x = ground_pt.x
        ground_pt_msg.y = ground_pt.y
        ground_pt_msg.z = ground_pt.z

        return ground_pt_msg

    def detections_cb(self, detections_msg):
        """
        Projects a list of bounding boxes on the ground reference frame point by point by
        calling :py:meth:`pixel_msg_to_ground_msg`. Then publishes the projected list of segments.

        Args:
            seglist_msg (:obj:`duckietown_msgs.msg.SegmentList`): Line segments in pixel space from
            unrectified images

        """
        def create_vector(x, y):
            vector = Vector2D()
            vector.x = x
            vector.y = y
            return vector
    
        if self.camera_info_received:
            detectionsList_out = SegmentList()
            #detectionsList_out.header = detections_msg.header
            #self.log(f"Received segments: {detections_msg}")

            bboxes = []
            for i in range(len(detections_msg.data) // 4):
                bbox = detections_msg.data[i * 4 : (i + 1) * 4]
                bboxes.append(bbox)

            for received_bbox in bboxes:
                
                ground_bbox = Segment()
                #x_center_norm, y_center_norm, width_norm, height_norm = received_bbox
                x_TL, y_TL, x_BR, y_BR = received_bbox
                # Find corners
                #center_norm = create_vector(x_center_norm, y_center_norm)
                # top_left_norm = create_vector((x_center_norm - width_norm / 2)/IMAGE_SIZE, (y_center_norm - height_norm / 2)/IMAGE_SIZE)
                # bottom_right_norm = create_vector((x_center_norm + width_norm / 2)/IMAGE_SIZE, (y_center_norm + height_norm / 2)/IMAGE_SIZE)
                top_left_norm = create_vector(x_TL/IMAGE_SIZE, y_TL/IMAGE_SIZE)
                bottom_right_norm = create_vector(x_BR/IMAGE_SIZE, y_BR/IMAGE_SIZE)

                # Center 
                ground_bbox.points[0] = self.pixel_msg_to_ground_msg(top_left_norm)
                ground_bbox.points[1] = self.pixel_msg_to_ground_msg(bottom_right_norm)
                #self.log(f"--------------------------------------->Recieved bbox: ({top_left_norm},{bottom_right_norm}) \n Projected bbox: {ground_bbox}")
                # TODO what about normal and points
                detectionsList_out.segments.append(ground_bbox)
            #self.log(f"Projected segments{ground_bbox}")
            self.pub_ground_detections.publish(detectionsList_out)

            if not self.first_processing_done:
                self.log("First projected segments published.")
                self.first_processing_done = True

            if self.pub_debug_img.get_num_connections() > 0:
                debug_image_msg = self.bridge.cv2_to_compressed_imgmsg(self.debug_image(detectionsList_out))
                debug_image_msg.header = detectionsList_out.header
                self.pub_debug_img.publish(debug_image_msg)
        else:
            self.log("Waiting for a CameraInfo message", "warn")

    def load_extrinsics(self):
        """
        Loads the homography matrix from the extrinsic calibration file.

        Returns:
            :obj:`numpy array`: the loaded homography matrix

        """
        # load intrinsic calibration
        cali_file_folder = "/data/config/calibrations/camera_extrinsic/"
        cali_file = cali_file_folder + rospy.get_namespace().strip("/") + ".yaml"

        # Locate calibration yaml file or use the default otherwise
        if not os.path.isfile(cali_file):
            self.log(
                f"Can't find calibration file: {cali_file}.\n Using default calibration instead.", "warn"
            )
            cali_file = os.path.join(cali_file_folder, "default.yaml")

        # Shutdown if no calibration file not found
        if not os.path.isfile(cali_file):
            msg = "Found no calibration file ... aborting"
            self.logerr(msg)
            rospy.signal_shutdown(msg)

        try:
            with open(cali_file, "r") as stream:
                calib_data = yaml.load(stream, Loader=yaml.Loader)
        except yaml.YAMLError:
            msg = f"Error in parsing calibration file {cali_file} ... aborting"
            self.logerr(msg)
            rospy.signal_shutdown(msg)

        #return calib_data["homography"]
        #return [0, -1, 0, 1, 0, 0, 0, 0, 1]
        return [-26.933, -63.336, 356.988, 309.672, -49.979, 259.2966, -0.0185, -0.2815, 1]

    def debug_image(self, detectionsList_out):
        """
        Generates a debug image with all the projected segments plotted with respect to the robot's origin.

        Args:
            seg_list (:obj:`duckietown_msgs.msg.SegmentList`): Line segments in the ground plane relative
            to the robot origin

        Returns:
            :obj:`numpy array`: an OpenCV image

        """
        # Duckietown Recommendations
            # dimensions of the image are 1m x 1m so, 1px = 2.5mm
            # the origin is at x=200 and y=300
            
        # if that's the first call, generate the background
        if self.debug_img_bg is None:
            # # initialize gray image
            #self.debug_img_bg = np.ones((400, 400, 3), np.uint8) * 128
            self.debug_img_bg = np.ones((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8) * 128
            # # draw vertical lines of the grid
            # for vline in np.arange(40, 361, 40):
            #     cv2.line(
            #         self.debug_img_bg, pt1=(vline, 20), pt2=(vline, 300), color=(255, 255, 0), thickness=1
            #     )

            # # draw the coordinates
            # cv2.putText(
            #     self.debug_img_bg,
            #     "-20cm",
            #     (120 - 25, 300 + 15),
            #     cv2.FONT_HERSHEY_PLAIN,
            #     0.8,
            #     (255, 255, 0),
            #     1,
            # )
            # cv2.putText(
            #     self.debug_img_bg,
            #     "  0cm",
            #     (200 - 25, 300 + 15),
            #     cv2.FONT_HERSHEY_PLAIN,
            #     0.8,
            #     (255, 255, 0),
            #     1,
            # )
            # cv2.putText(
            #     self.debug_img_bg,
            #     "+20cm",
            #     (280 - 25, 300 + 15),
            #     cv2.FONT_HERSHEY_PLAIN,
            #     0.8,
            #     (255, 255, 0),
            #     1,
            # )

            # # draw horizontal lines of the grid
            # for hline in np.arange(20, 301, 40):
            #     cv2.line(
            #         self.debug_img_bg, pt1=(40, hline), pt2=(360, hline), color=(255, 255, 0), thickness=1
            #     )

            # # draw the coordinates
            # cv2.putText(
            #     self.debug_img_bg, "20cm", (2, 220 + 3), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 0), 1
            # )
            # cv2.putText(
            #     self.debug_img_bg, " 0cm", (2, 300 + 3), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 0), 1
            # )

            # # draw robot marker at the center
            # cv2.line(
            #     self.debug_img_bg,
            #     pt1=(200 + 0, 300 - 20),
            #     pt2=(200 + 0, 300 + 0),
            #     color=(255, 0, 0),
            #     thickness=1,
            # )

            # cv2.line(
            #     self.debug_img_bg,
            #     pt1=(200 + 20, 300 - 20),
            #     pt2=(200 + 0, 300 + 0),
            #     color=(255, 0, 0),
            #     thickness=1,
            # )

            # cv2.line(
            #     self.debug_img_bg,
            #     pt1=(200 - 20, 300 - 20),
            #     pt2=(200 + 0, 300 + 0),
            #     color=(255, 0, 0),
            #     thickness=1,
            # )

        image = self.debug_img_bg.copy()
        
        # plot every segment if both ends are in the scope of the image (within 50cm from the origin)
        #self.log(f"-------------------------------------->Projected segments: {detectionsList_out}")
        for segment in detectionsList_out.segments:
            # if not np.any(
            #     np.abs([segment.points[0].x, segment.points[0].y, segment.points[1].x, segment.points[1].y])
            #     > 0.50
            # ):
            # cv2.rectangle(image, 
            #                  pt1=(int((segment.points[0].y) * -400) + 200, int(segment.points[0].x * -400) + 300),
            #                  pt2=(int(segment.points[1].y * -400) + 200, int(segment.points[1].x * -400) + 300),
            #                  color=(255, 0, 0), 
            #                  thickness=1)
            #self.log(f"-------------------------------------->Projected bbox: {segment}")
            pt1 = (int(abs(segment.points[0].x * -IMAGE_SIZE)), int(abs(segment.points[0].y * IMAGE_SIZE)))
            pt2 = (int(abs(segment.points[1].x * -IMAGE_SIZE)), int(abs(segment.points[1].y * IMAGE_SIZE)))
            cv2.rectangle(image, 
                             pt1=pt1,
                             pt2= pt2,
                             color=(255, 0, 0), 
                             thickness=1)
            #self.log(f"-------------------------------------->Projected bbox: {pt1}, {pt2}")
        return image





if __name__ == "__main__":
    projection_node = ProjectionNode(node_name="projection_node")
    rospy.spin()