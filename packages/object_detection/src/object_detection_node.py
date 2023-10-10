#!/usr/bin/env python3

import cv2
import math
import numpy as np
import rospy
from typing import Optional, Tuple

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
        self.v = 0.2
        aido_eval = rospy.get_param("~AIDO_eval", False)
        self.log(f"AIDO EVAL VAR: {aido_eval}")
        self.log("Starting model loading!")
        
        # Load Yolo model
        self._debug = rospy.get_param("~debug", False)
        self.model_wrapper = Wrapper(aido_eval)
        self.log("Finished model loading!")
        self.frame_id = 0
        self.first_image_received = False

        # Initialise Braitenburg constants
        self.gain: float = 10.0
        self.omega: float = 0.0
        self.l_max = -math.inf
        self.r_max = -math.inf
        self.l_min = math.inf
        self.r_min = math.inf
        self.left  = None
        self.right = None

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
            self.pub_car_commands(self.omega, image_msg.header)
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


        # Only move for a valid detection
        if detection:
            self.log("Valid Detection")
            map_bare = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)

            for clas, box in zip(classes, bboxes):
                pt1 = np.array([int(box[0]), int(box[1])])
                pt2 = np.array([int(box[2]), int(box[3])])

                pt1 = tuple(pt1)
                pt2 = tuple(pt2)

                color = (0, 0, 255)
                # draw bounding box
                map_bbox = cv2.rectangle(map_bare, pt1, pt2, color, thickness = -1)


            # Wheel commands
            map = map_bbox[:, :, 2]  # Index 0 corresponds to the red channel
            sum = np.sum(map)
            self.log(f"Sum: {sum}")
            self.omega = self.compute_commands(map)
            self.pub_car_commands(self.omega, image_msg.header)
            

            # Publish image
            map_bgr = map_bbox[..., ::-1]
            weight_img = self.bridge.cv2_to_compressed_imgmsg(map_bgr)
            self.pub_debug_img.publish(weight_img)
        else:
            self.pub_car_commands(self.omega, image_msg.header)
        
        # Publish debug image
        if self._debug:
            colors = {0: (0, 255, 255), 1: (0, 165, 255), 2: (0, 250, 0), 3: (0, 0, 255), 4: (255, 0, 0)}
            names = {0: "duckie", 1: "cone", 2: "truck", 3: "bus", 4: "duckiebot"}
            font = cv2.FONT_HERSHEY_SIMPLEX
            for clas, box in zip(classes, bboxes):
                pt1 = np.array([int(box[0]), int(box[1])])
                pt2 = np.array([int(box[2]), int(box[3])])

                pt1 = tuple(pt1)
                pt2 = tuple(pt2)

                color = tuple(reversed(colors[clas]))
                name = names[clas]

                # label location
                text_location = (pt1[0], min(pt2[1] + 30, IMAGE_SIZE))
                rgb = cv2.rectangle(rgb, pt1, pt2, color, thickness = 2)
                rgb = cv2.putText(rgb, name, text_location, font, 1, color, thickness=2)

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
    
    def compute_commands(self, map) -> float:
        """Returns the commands (pwm_left, pwm_right)"""

        # If we have not received any image, we don't move
        if map is None:
            return 0.0

        if self.left is None:
            # if it is the first time, we initialize the structures
            shape = map.shape[0], map.shape[1]
            self.left = self.get_motor_left_matrix(shape)
            self.right = self.get_motor_right_matrix(shape)

        # now we just compute the activation of our sensors
        l = float(np.sum(map * self.left))
        r = float(np.sum(map * self.right))

        self.log(f"Before normalization: {l}, {r}")
        # These are big numbers -- we want to normalize them.
        # We normalize them using the history

        # first, we remember the high/low of these raw signals
        self.l_max = max(l, self.l_max)
        self.r_max = max(r, self.r_max)
        self.l_min = min(l, self.l_min)
        self.r_min = min(r, self.r_min)


        # now rescale from 0 to 1
        ls = self.rescale(l, self.l_min, self.l_max)
        rs = self.rescale(r, self.r_min, self.r_max)
    
        gain = self.gain
        left =  ls * gain
        right = -rs * gain

        self.log(f"ls: {ls}, rs: {rs}")
        self.omega = left+right
        self.log(f"omega: {self.omega}")
        return self.omega
    
    def pub_car_commands(self, omega, header):
        # TODO: must change this publish pwm to each wheel
        car_control_msg = Twist2DStamped()
        car_control_msg.header = header

        # always drive straight
        car_control_msg.v = self.v

        # Turn based on bratienburg
        car_control_msg.omega = omega

        self.pub_car_cmd.publish(car_control_msg)
    

    def rescale(self, a: float, L: float, U: float):
        if np.allclose(L, U):
            return 0.0
        return (a - L) / (U - L)
    
    def get_motor_left_matrix(self, shape: Tuple[int, int]) -> np.ndarray:
        res = np.zeros(shape=shape, dtype="float32")
        res[:, :int(shape[1]/2)] = 1

        return res


    def get_motor_right_matrix(self, shape: Tuple[int, int]) -> np.ndarray:

        res = np.zeros(shape=shape, dtype="float32")
        # these are random values
        res[:, int(shape[1]/2):] = 1

        return res
    
if __name__ == "__main__":
    # Initialize the node
    object_detection_node = ObjectDetectionNode(node_name="object_detection_node")
    # Keep it spinning
    rospy.spin()
