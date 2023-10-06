#!/usr/bin/env python3

import cv2
import os
import yaml
import numpy as np
import rospy

from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import Vector2D, Twist2DStamped, EpisodeStart
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point as PointMsg

from image_processing.utils import get_camera_info_default
from image_processing.rectification import Rectify
from image_processing.ground_projection_geometry import GroundProjectionGeometry, Point
from nn_model.model import Wrapper
from nn_model.constants import IMAGE_SIZE
from nn_model.constants import \
    NUMBER_FRAMES_SKIPPED, \
    filter_by_classes, \
    filter_by_bboxes, \
    filter_by_scores


class ObjectDetectionNode(DTROS):
    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(ObjectDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        self.initialized = False
        self.log("Initializing!")

        # Get vehicle name
        self.veh = rospy.get_namespace().strip("/")
        self.avoid_duckies = False
        self.width_limit = 20 

        # Construct publishers
        car_cmd_topic = f"/{self.veh}/joy_mapper_node/car_cmd"
        self.pub_car_cmd = rospy.Publisher(
            car_cmd_topic,
            Twist2DStamped,
            queue_size=1,
            dt_topic_type=TopicType.CONTROL
        )

        self.pub_detections_image = rospy.Publisher(
            "~image/compressed",
            CompressedImage,
            queue_size=1,
            dt_topic_type=TopicType.DEBUG
        )

        self.pub_debug_img = rospy.Publisher(
            "~debug/ground_projection_image/compressed",
            CompressedImage,
            queue_size=1,
            dt_topic_type=TopicType.DEBUG,
        )

        # Construct subscribers
        episode_start_topic = f"/{self.veh}/episode_start"
        rospy.Subscriber(
            episode_start_topic,
            EpisodeStart,
            self.cb_episode_start,
            queue_size=1
        )

        self.sub_image = rospy.Subscriber(
            f"/{self.veh}/camera_node/image/compressed",
            CompressedImage,
            self.image_cb,
            buff_size=10000000,
            queue_size=1,
        )

        self.bridge = CvBridge()
        # Configure AIDO
        self.v = rospy.get_param("~speed", 0.0)
        aido_eval = rospy.get_param("~AIDO_eval", False)
        self.log(f"AIDO EVAL VAR: {aido_eval}")
        self.log("Starting model loading!")
        
        # Load Yolo model
        self._debug = rospy.get_param("~debug", False)
        self.model_wrapper = Wrapper(aido_eval)
        self.log("Finished model loading!")
        self.frame_id = 0
        self.first_image_received = False

        # Initialise Ground projection utils
        self.ground_projector = None
        self.debug_img_bg = None
        self.rectifier = None
        self.homography = self.load_extrinsics()
        self.log(f"Loaded homography matrix: {np.array(self.homography).reshape((3, 3))}")
        self.camera_info = get_camera_info_default()
        self.camera_info_received = True
        self.log("loaded camera info")
        self.rectifier = Rectify(self.camera_info)
        self.ground_projector = GroundProjectionGeometry(
            im_width=IMAGE_SIZE, im_height=IMAGE_SIZE, homography=np.array(self.homography).reshape((3, 3)))
        self.log("CameraInfo received.")
        self.camera_info_received = True

        self.first_processing_done = False

        self.initialized = True
        self.log("Initialized!")

    def cb_episode_start(self, msg: EpisodeStart):
        self.log("Episode started")
        self.avoid_duckies = False
        self.pub_car_commands(True, msg.header)

    def image_cb(self, image_msg):
        if not self.initialized:
            self.pub_car_commands(True, image_msg.header)
            return

        # Only call Yolo model after user specified frames
        self.frame_id += 1
        self.frame_id = self.frame_id % (1 + NUMBER_FRAMES_SKIPPED())

        if self.frame_id != 0:
            self.pub_car_commands(self.avoid_duckies, image_msg.header)
            return

        # Decode from compressed image with OpenCV
        try:
            bgr = self.bridge.compressed_imgmsg_to_cv2(image_msg)
        except ValueError as e:
            self.logerr("Could not decode image: %s" % e)
            return

        # Convert from BGR to RGB for model
        rgb = bgr[..., ::-1]

        # Resize for model
        rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))

        # YOLOv5 prediction
        bboxes, classes, scores = self.model_wrapper.predict(rgb)
        detection = self.det2bool(bboxes, classes, scores)

        # Only project if a valid detection
        if detection:
            self.log("Valid Detection")
            self.projection(bboxes, image_msg.header)

        else:
            self.pub_car_commands(self.avoid_duckies, image_msg.header)

        if self._debug:
            colors = {0: (0, 255, 255), 1: (0, 165, 255), 2: (0, 250, 0), 3: (0, 0, 255), 4: (255, 0, 0)}
            names = {0: "duckie", 1: "cone", 2: "truck", 3: "bus", 4: "duckiebot"}
            font = cv2.FONT_HERSHEY_SIMPLEX
            for clas, box in zip(classes, bboxes):
                pt1 = np.array([int(box[0]), int(box[1])])
                pt2 = np.array([int(box[2]), int(box[3])])
                #height = int(box[3]-box[1])
                #distance = int(34.1549*np.exp(height*-0.0275283))
                pt1 = tuple(pt1)
                pt2 = tuple(pt2)
                color = tuple(reversed(colors[clas]))
                name = names[clas]
                # draw bounding box
                rgb = cv2.rectangle(rgb, pt1, pt2, color, 2)
                # label location
                text_location = (pt1[0], min(pt2[1] + 30, IMAGE_SIZE))
                #distance_location = (pt1[0], min(pt2[1] + 60, IMAGE_SIZE))
                # if distance < 0:
                #     distance = 0
                #draw label underneath the bounding box
                rgb = cv2.putText(rgb, name, text_location, font, 1, color, thickness=2)
                # if distance < 20:
                #     distance =  cv2.putText(rgb, str(distance), text_location, font, 1, color, thickness=2)

            # Publish detection debug image
            bgr = rgb[..., ::-1]
            obj_det_img = self.bridge.cv2_to_compressed_imgmsg(bgr)
            self.pub_detections_image.publish(obj_det_img)


    def det2bool(self, bboxes, classes, scores):
        box_ids = np.array(list(map(filter_by_bboxes, bboxes))).nonzero()[0]
        cla_ids = np.array(list(map(filter_by_classes, classes))).nonzero()[0]
        sco_ids = np.array(list(map(filter_by_scores, scores))).nonzero()[0]

        box_cla_ids = set(list(box_ids)).intersection(set(list(cla_ids)))
        box_cla_sco_ids = set(list(sco_ids)).intersection(set(list(box_cla_ids)))

        if len(box_cla_sco_ids) > 0:
            return True
        else:
            return False
    
    def pub_car_commands(self, stop, header):
        car_control_msg = Twist2DStamped()
        car_control_msg.header = header

        if stop:
            car_control_msg.v = 0.0
        else:
            car_control_msg.v = self.v

        # always drive straight
        car_control_msg.omega = 0.0

        self.pub_car_cmd.publish(car_control_msg)
    
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

        ground_pt = self.ground_projector.pixel2ground(norm_pt)
        # point to message
        ground_pt_msg = PointMsg()
        ground_pt_msg.x = ground_pt.x
        ground_pt_msg.y = ground_pt.y
        ground_pt_msg.z = ground_pt.z

        return ground_pt_msg
    
    def projection(self, bboxes, header):

        def create_vector(x, y):
            vector = Vector2D()
            vector.x = x
            vector.y = y
            return vector
        
        projected_bboxes = []

        if self.camera_info_received:

            for received_bbox in bboxes:
                x_TL, y_TL, x_BR, y_BR = received_bbox

                # Normalise the detected corners
                top_left_norm = create_vector(x_TL/IMAGE_SIZE, y_TL/IMAGE_SIZE)
                bottom_right_norm = create_vector(x_BR/IMAGE_SIZE, y_BR/IMAGE_SIZE)

                # Project coners
                top_left_proj = self.pixel_msg_to_ground_msg(top_left_norm)
                bottom_right_proj = self.pixel_msg_to_ground_msg(bottom_right_norm)

                # Width of projected bbox
                width = abs(top_left_proj.x - bottom_right_proj.x)*IMAGE_SIZE
                self.log(f"Width: {width}")
                if width > self.width_limit: 
                    self.log(f"Duckie pedestrian detected... stopping: {width} > {self.width_limit}")
                    self.avoid_duckies = True

                # Store the projected bounding box as a tuple (x_TL, y_TL, x_BR, y_BR)
                projected_bboxes.append((top_left_proj.x, top_left_proj.y, bottom_right_proj.x, bottom_right_proj.y))

            if not self.first_processing_done:
                self.log("First projected segments published.")
                self.first_processing_done = True

            # Generate debug image
            debug_image_msg = self.bridge.cv2_to_compressed_imgmsg(self.debug_image(projected_bboxes))
            debug_image_msg.header = header
            self.pub_debug_img.publish(debug_image_msg)

            # Publish wheel commands
            self.pub_car_commands(self.avoid_duckies, header)

        else:
            self.log("Waiting for a CameraInfo message", "warn")


    def debug_image(self, projected_bboxes):
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
            self.debug_img_bg = np.ones((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8) * 128

        image = self.debug_img_bg.copy()
        
        for bbox in projected_bboxes:
            x_TL_proj, y_TL_proj, x_BR_proj, y_BR_proj = bbox

            # Calculate pt1 and pt2 based on the projected points
            pt1 = (int(abs(x_TL_proj * -IMAGE_SIZE)), int(abs(y_TL_proj * IMAGE_SIZE)))
            pt2 = (int(abs(x_BR_proj * -IMAGE_SIZE)), int(abs(y_BR_proj * IMAGE_SIZE)))

            cv2.rectangle(image, 
                             pt1=pt1,
                             pt2= pt2,
                             color=(255, 0, 0), 
                             thickness=1)
        return image

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
        return [0, -1, 0, 1, 0, 0, 0, 0, 1]
        #return [-26.933, -63.336, 356.988, 309.672, -49.979, 259.2966, -0.0185, -0.2815, 1]
    
if __name__ == "__main__":
    # Initialize the node
    object_detection_node = ObjectDetectionNode(node_name="object_detection_node")
    # Keep it spinning
    rospy.spin()
