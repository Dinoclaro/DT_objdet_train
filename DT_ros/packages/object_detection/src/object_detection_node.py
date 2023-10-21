#!/usr/bin/env python3

import cv2
import math
import numpy as np
import rospy
import time
from typing import Optional, Tuple

from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import  Twist2DStamped, EpisodeStart, WheelsCmdStamped
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
        self.gain: float = 1 # 3
        self.const: float = 0.3 # 0.3
        self.l_max = -math.inf
        self.r_max = -math.inf
        self.l_min = math.inf
        self.r_min = math.inf
        self.left  = None
        self.right = None


        # Construct publishers
        wheels_cmd_topic = f"/{self.veh}/wheels_driver_node/wheels_cmd"
        self.pub_car_cmd = rospy.Publisher(
            wheels_cmd_topic,
            WheelsCmdStamped,
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
        self._debug = rospy.get_param("~debug", True)
        self.model_wrapper = Wrapper(aido_eval)
        self.log("Finished model loading!")
        self.frame_id = 0
        self.count = 0

        # Choose Bratienburg config 
        self.type = 0 # 0 for FEAR, 1 for EXPLORE
        self.weight = 1  # 0 for Basic, 1 half, 2 for half and triangle
        self.first_image_received = False

        self.first_processing_done = False

        self.initialized = True
        self.log("Initialized!")

    def cb_episode_start(self, msg: EpisodeStart):
        self.log("Episode started")
        self.avoid_duckies = False
        self.pub_car_commands(self.const, self.const, msg.header)

    def image_cb(self, image_msg):
        if not self.initialized:
            #self.pub_car_commands(self.const, self.const, image_msg.header)
            return

        # Only call Yolo model after user specified frames
        self.frame_id += 1
        self.frame_id = self.frame_id % (1 + NUMBER_FRAMES_SKIPPED())

        if self.frame_id != 0:
            # self.pub_car_commands(self.const, self.const, image_msg.header)
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
        start_time = time.time()

        bboxes, classes, scores = self.model_wrapper.predict(rgb)

        yolo_time = time.time()
        execution_time = yolo_time - start_time
        self.log(f"YOLO time: {execution_time}")

        detection = self.det2bool(bboxes, classes, scores)


        # Only move for a valid detection
        if detection:
            #self.log("Valid Detection")
            map_bare = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)

            for clas, box in zip(classes, bboxes):
                # TODO: Make this only for duckies above the minimum bbox size
                pt1 = np.array([int(box[0]), int(box[1])])
                pt2 = np.array([int(box[2]), int(box[3])])

                pt1 = tuple(pt1)
                pt2 = tuple(pt2)

                color = (0, 0, 255)
                # draw bounding box
                map_bbox = cv2.rectangle(map_bare, pt1, pt2, color, thickness = 1)


            # Wheel commands
            map = map_bbox[:, :, 2]  # Index 0 corresponds to the red channel
            vel = self.compute_commands(map)
            comm_time = time.time()
            exec_comm_time = comm_time -start_time
            self.log(f"Wheel command time: {exec_comm_time}")
            self.pub_car_commands(vel[0], vel[1], image_msg.header)
            

            # Publish image
            map_bgr = map_bbox[..., ::-1]
            weight_img = self.bridge.cv2_to_compressed_imgmsg(map_bgr)
            self.pub_debug_img.publish(weight_img)
        # else:
        #     self.pub_car_commands(self.const, self.const, image_msg.header)
        
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
                
                rgb = cv2.rectangle(rgb, pt1, pt2, color, thickness = 2)

                # # label location
                # text_location = (pt1[0], min(pt2[1] + 30, IMAGE_SIZE))

                # rgb = cv2.putText(rgb, name, text_location, font, 1, color, thickness=2)

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
    
    def compute_commands(self, map) -> Tuple[float, float]:
        """Returns the commands (pwm_left, pwm_right)"""

        # If we have not received any image, we don't move
        if map is None:
            return 0.0

        if self.left is None:
            # if it is the first time, we initialize the structures
            shape = map.shape[0], map.shape[1]
            self.left = self.get_motor_left_matrix(shape)[0]
            self.l_max = self.get_motor_left_matrix(shape)[1]

            self.right = self.get_motor_right_matrix(shape)[0]
            self.r_max = self.get_motor_right_matrix(shape)[1]

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
        self.count += 1
        
        self.log(f"count: {self.count}")
    
        # now rescale from 0 to 1
        ls = self.rescale(l, self.l_min, self.l_max)
        rs = self.rescale(r, self.r_min, self.r_max)
        self.log(f"after rescale: {ls}, {rs} \n max: {self.l_max}, {self.r_max} \n min: {self.l_min}, {self.r_min}")

        pwm_left = self.const - rs * self.gain
        pwm_left = self.rescale(pwm_left, 0, (self.gain+self.const))    # Max the pwm can be is (1*gain + const)
        
        pwm_right = self.const - ls * self.gain
        pwm_right = self.rescale(pwm_right, 0, (self.gain+self.const))  # Max the pwm can be is (1*gain + const)
        
        # pwm_left = self.const + ls * self.gain
        # pwm_left = self.rescale(pwm_left, 0, (self.gain+self.const))    # Max the pwm can be is (1*gain + const)
        
        # pwm_right = self.const + rs * self.gain
        # pwm_right = self.rescale(pwm_right, 0, (self.gain+self.const))  # Max the pwm can be is (1*gain + const)

        self.log(f"ls: {pwm_left}, rs: {pwm_right}")

        return pwm_left, pwm_right
    
    def pub_car_commands(self, pwm_left, pwm_right, header):
        wheel_control_msg = WheelsCmdStamped()
        wheel_control_msg.header = header

        # always drive straight
        wheel_control_msg.vel_left = pwm_left
        self.log(f"vel_left: {wheel_control_msg.vel_left}")

        # Turn based on bratienburg
        wheel_control_msg.vel_right = pwm_right
        self.log(f"vel_right: {wheel_control_msg.vel_right}")

        self.pub_car_cmd.publish(wheel_control_msg)
    

    def rescale(self, a: float, L: float, U: float):
        if np.allclose(L, U):
            return 0.0
        elif a > U:
            return 1.0
        
        return (a - L) / (U - L)
    
    def get_motor_left_matrix(self, shape: Tuple[int, int]) -> np.ndarray:
        res = np.zeros(shape=shape, dtype="float32")

        if self.weight == 0:
            l_max = 0
            # l_max = 2000000.0
            res[:, :int(shape[1]/2)] = 1

        elif self.weight == 1:
            l_max = 100000.0
            l_max = 0
            half= int(shape[1]/2)
            width = 250
            res[width: , : half] = 1

        elif self.weight == 2:
            l_max = 0
            # l_max = 994500.0
            # Define the vertices of the triangle
            vertices = np.array([[0, shape[1]], [shape[1]/2, shape[1]], [shape[1]/2, 150]], np.int32)

            # Reshape vertices to fit the fillPoly function
            vertices = vertices.reshape((-1, 1, 2))
            # Fill the triangle with a color (in this case, red)
            color = (1)
            cv2.fillPoly(res, [vertices], color)# Define the vertices of the triangle


        return res, l_max

    def get_motor_right_matrix(self, shape: Tuple[int, int]) -> np.ndarray:

        res = np.zeros(shape=shape, dtype="float32")
        
        if self.weight == 0:
            # r_max = 2000000.0
            r_max = 0 
            res[:, int(shape[1]/2):] = 1

        elif self.weight == 1:
            r_max = 100000.0
            r_max = 0 
            half= int(shape[1]/2)
            width = 250
            res[width: ,half :] = 1
            
        elif self.weight == 2:
            r_max = 0 
            # r_max = 994500.0
            # Define the vertices of the triangle
            vertices = np.array([[shape[1], shape[1]], [shape[1]/2, shape[1]], [shape[1]/2, 150]], np.int32)

            # Reshape vertices to fit the fillPoly function
            vertices = vertices.reshape((-1, 1, 2))

            # Fill the triangle with a color (in this case, red)
            color = (1)
            cv2.fillPoly(res, [vertices], color)# Define the vertices of the triangle

        return res, r_max
    
if __name__ == "__main__":
    # Initialize the node
    object_detection_node = ObjectDetectionNode(node_name="object_detection_node")
    # Keep it spinning
    rospy.spin()
