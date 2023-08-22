#!/usr/bin/env python3

import cv2
import numpy as np
import rospy

from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import Twist2DStamped, EpisodeStart
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

from nn_model.constants import IMAGE_SIZE
from nn_model.model import Wrapper

from utils import \
    NUMBER_FRAMES_SKIPPED, \
    filter_by_classes, \
    filter_by_boxes, \
    filter_by_scores

class OD_node(DTROS):

    # --------------------
    # Constructor Method
    # --------------------
    def __init__(self, node_name):
        # Initialize the DTROS parent class with PERCEPTION node type 
        super(OD_node, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        self.initialized = False # Check if node has been properly initialised 
        self.log("Initialising OD_node.")

        # Retrieve namespace of ROS node and remove any slashes
        self.veh = rospy.get_namespace().strip("/")
        # DT configuration, initialise class attribute as false 
        self.avoid_duckies = False

        # --------------------
        # Construct Publisher
        # --------------------

        # CAR COMMAND: Initialise ROS publisher for publishing control commands to 'car_cmd_topic'

        #   Topic name: car_cmd_topic
        #   Message Type: Twist2DStamped
        #   Queue Size: 1 -> Only latest message should be kept
        #   Topic Type: CONTROL
        car_cmd_topic = f"/{self.veh}/joy_mapper_node/car_cmd"
            # 'Twist2DStamped' message type associated with topic type CONTROL
        self.pub_car_cmd = rospy.Publisher(car_cmd_topic, 
                                           Twist2DStamped,
                                           queue_size = 1,
                                           dt_topic_type=TopicType.Control)
        
        # COMPRESSED_IMAGES: publishing compressed images related to detected objects 

        #   Topic name: "~image/compressed"
        #   Message Type: CompressedImage
        #   Queue Size: 1 -> Only latest message should be kept
        #   Topic Type: DEBUG
        self.pub_detection_images = rospy.Publisher("~image/compressed",
                                                    CompressedImage,
                                                    queue_size = 1,
                                                    dt_topic_type = TopicType.DEBUG)

        # --------------------
        # Construct Subscriber
        # --------------------
        
        # EPISODE START: Subscriber for listening to episode_start_topic

        episode_start_topic = f"/{self.veh}/episode_start"  # Construct topic name for subscribing to episode
 
        #   Topic name to listen to: episode_start_topic
        #   Message Type: EpisodeStart
        #   Callback Function: self.cb_episode_start
        #   Queue Size: 1 -> Only latest message should be kept
        rospy.Subscriber(episode_start_topic,
                         EpisodeStart,
                         self.cb_episode_start,
                         queue_size = 1)
        
        # COMPRESSED_IMAGE: receiving compressed image data from camera topic 
        
        #   Topic name to  listen to: f"/{self.veh}/camera_node/image/compressed"
        #   Message Type: CompressedImage
        #   Callback Function: self.image_cb 
        #   Buff Size: 10MB buffer size for incoming messages 
        #   Queue Size: 1 -> Only latest message should be kept
        self.sub_image = rospy.Subscriber(f"/{self.veh}/camera_node/image/compressed",
                                          CompressedImage,
                                          self.image_cb,
                                          buff_size = 10000000,
                                          queue_size = 1,)
        
        # --------------------
        # Configuration
        # --------------------

        # ROS utility to convert ROS image messages and OpenCV images
        self.bridge = CvBridge()

        # Duckinator speed: default = 0.0
        self.v = rospy.get_param("~speed", 0.0)

        # Load model
        self.log("Model loading!")
        self.debug = rospy.get_param("~debug", False)   # Disable debug mode
        self.log("Finished Loading!")
        self.frame_id = 0 # Initialises image ID to 0 
        self.first_image_received = False # Image not received yet 
        self.initialized = True
        self.log("Initialized!")

    # --------------------
    # Methods:
    # --------------------
        
    def cb_episode_start(self, msg: EpisodeStart):
        # callback function executed when an "episode_start" message of type 'EpisodeStart' is received
        self.avoid_duckies = False 
        self.pub_car_cmd(True, msg.header)  # Initiate behaviour at start of an episode
    
    def image_cb(self, image_msg):
        # image_cb: invoked when a compressed image message is received from subscriber topic.

        # Stop further processing of image if not fully initialized
        if not self.initialized:
            self.pub_car_cmd(True, image_msg.header)
            return  
        
        # Check if should execute received image
        self.frame_id += 1
        self.frame_id = self.frame_id % (1 + NUMBER_FRAMES_SKIPPED())
        if self.frame_id != 0:
            self.pub_car_cmd(self.avoid_duckies, image_msg.header)
            return 

        # Decode compressed image with OpenCv
        try:
            bgr = self.bridge.compressed_imgmsg_to_cv2(image_msg)
        except ValueError as e:
            self.logger("Could not decode: %s" % e)
            return 
        
        # Convert OpenCV BGR to RGB
        rgb = bgr[..., ::-1] # Elements on the along the last axis should be reversed

        # Resize image for Yolov5 compatibility
        rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))

        # Create a model wrapper used for prediction
        bboxes, classes, scores = self.model_wrapper.predict(rgb)

        # Detection Check:
        detection = self.det2bool(bboxes, classes, scores)   # Convert to boolean value 
        if detection:
            self.log("Duckie pedestrian detected ... Stopping")
            self.avoid_duckies = True 
        # Publish car command with avoid_duckies flag set to true 
        self.pub_car_cmd(self.avoid_duckies, image_msg.header)

        # Debug Visualisation: Bounding boxes and labels are drawn for each object detected 
        if self._debug:
            # Box and label set-up: a separate colour for each bounding box
            colours = {0: (0, 255, 255), 1: (0, 165, 255), 2: (0, 250, 0), 3: (0, 0, 255)}
            names = {0: "duckie", 1: "cone", 2: "truck", 3: "bus"}
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Draw box and labels only for objects chosen in utils.py
            for clas, box in zip(classes, bboxes):
                p1 = np.array([int(box[0]), int(box[1])])   # Top-left corner 
                p2 = np.array([int(box[2]), int(box[3])])   # Bottom-right corner 
                p1 = tuple(p1)  # Tuples used in OpenCV
                p2 = tuple(p2)

                color = tuple(reversed(colours[clas])) # Retrieve bbox colour for object 
                name = names[clas]
                
                # Draw bbox
                rgb = cv2.rectangle(rgb, p1, p2, color, 2) # line thickness of 2
                # Insert label
                    # x-cord: top-left corner of bbox
                    # y-cord: 30 pixels below bottom right of hte corner
                location = (p1[0], min(p2[1 + 30, IMAGE_SIZE]))
                rgb = cv2.puttext(rgb, name, location, font, 1, color, thickness = 2) # Font size 1
            
            bgr = rgb[..., ::-1] # Convert back to bgr
            od_image = self.bridge.cv2_to_compressed_imgmsg(bgr)     # Convert bgr to compressed image message of type 'CompressedImage'
            self.pub_detection_images.publish(od_image) 
    
    def det2bool(self, bboxes, classes, scores):
        """
        Determines whether detection is valid based on constraints in utils.py

        Args:
            bboxes: minimum area of bounding box for valid detection 
            classes: valid classes 
            scores: minimum valid confidence

        Return: 
            bool: True if detection is valid, False otherwise
        """
        # Maps the filter functions defined in utils.py over the results of the YOLO model to determine if it is a valid detection
        box_ids = np.array(list(map(filter_by_boxes, bboxes))).nonzero()[0]
        cls_ids = np.array(list(map(filter_by_classes, classes))).nonzero()[0]
        sco_ids = np.array(list(map(filter_by_scores, scores))).nonzero()[0]

        #  Find if there are any intersections
        box_cla_int = set(list(box_ids)).intersection(set(list(cls_ids)))
        box_cla_sco_int = set(list(sco_ids)).intersection(set(list(box_cla_int)))

        if len(box_cla_sco_int) > 0:
            return True
        else: 
            return False
        
    def pub_car_commands(self, stop, header):
        """
        Publishes car commands to the 'car_cmd_topic' topic

        Args:
            stop: boolean value to determine if car should stop
            header: header of the image message
        """

        # Creates an instance of 'Twist2DStamped' message type
        car_control_msg = Twist2DStamped()  
        
        # ensures proper co-ordination and synchronisation of messages
        car_control_msg.header = header 

        # Control duckiebot
        if stop:
            car_control_msg.v = 0.0
        else:
            car_control_msg.v = self.v

        # Drive straight 
        car_control_msg.omega = 0.0

        self.pub_car_cmd.publish(car_control_msg)


if __name__ == "__main__":
    # Intialise the node
    object_detection_node = OD_node(node_name="object_detection_node")

    # Ensure that the node stays active and continues to interact with the ROS ecosystem
    rospy.spin()