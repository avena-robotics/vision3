import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import QoSDurabilityPolicy,QoSProfile,QoSReliabilityPolicy,QoSHistoryPolicy
import cv2
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from cv_bridge import CvBridge
from numba.cuda import jit
# from ament_index_python.packages import get_package_share_directory

import time




class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('rectify')
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        self.left_image_subscription = self.create_subscription(
            Image,
            'left/image_raw',
            self.left_image_callback,
            qos)
        self.right_image_subscription = self.create_subscription(
            Image,
            'right/image_raw',
            self.right_image_callback ,
            qos)
        self.left_rect_publisher = self.create_publisher(Image, '/left/image_rect', qos)
        self.right_rect_publisher = self.create_publisher(Image, '/right/image_rect', qos)
        # pkg_share_directory = get_package_share_directory('basler_ros2_driver')
        pkg_share_directory = "/home/avena/nvidia_ws/src/basler_ros2_driver"
        # config_directory = pkg_share_directory + "/config/"
        config_directory = pkg_share_directory + "/config/"
        calibration_json_path =config_directory + "fhd_basler_calibration.json"
        rectification_json_path = config_directory +"fhd_basler_rectification.json"
        self.height =1080
        self.width =1920
        self.stereoMapL_x,self.stereoMapL_y = self.get_rectify_map(calibration_json_path=calibration_json_path,camera_id="cam1_ir1",rectification_json_path=rectification_json_path)
        self.stereoMapR_x, self.stereoMapR_y = self.get_rectify_map(calibration_json_path=calibration_json_path,camera_id="cam1_ir2",rectification_json_path=rectification_json_path)
        self.bridge = CvBridge()

    def read_custom_json(self,path: str = "") -> dict:
        with open(path, 'r') as f:
            return json.load(f)

    def read_camera_matrix(self,input_dict: dict, camera_id: str):
        temp = input_dict["intrinsic"][camera_id]
        template = np.eye(3)
        template[0][0] = temp["fx"]
        template[1][1] = temp["fy"]
        template[0][2] = temp["cx"]
        template[1][2] = temp["cy"]
        return template


    def read_distortion_vector(self,input_dict: dict, camera_id: str):
        template = input_dict["intrinsic"][camera_id]
        temp = np.array([template["k1"], template["k2"], template["p1"], template["p2"], template["k3"]])
        return temp


    def read_extrinsic(self,input_dict: dict, parent_id="", child_id=""):
        temp = input_dict["extrinsic"]
        for ext in temp:
            # Check whether there is requested transformation in input
            if ext["parent"] == parent_id and ext["child"] == child_id:
                quat = np.array([ext["R"]["x"], ext["R"]["y"], ext["R"]["z"], ext["R"]["w"]])
                rot = R.from_quat(quat)
                rot = rot.as_matrix()
                trans = np.array([ext["T"]["x"], ext["T"]["y"], ext["T"]["z"]])
                return np.array(rot.tolist()), trans
        return None, None
    
    def get_rectify_map(self,calibration_json_path: str = "/home/avena/nvidia_ws/src/basler_ros2_driver/config/fhd_basler_calibration.json",
            camera_id: str = "cam1_ir1",rectification_json_path: str = "/home/avena/nvidia_ws/src/basler_ros2_driver/config/fhd_basler_rectification.json"):
        calibration_json = self.read_custom_json(calibration_json_path)
        K = self.read_camera_matrix(calibration_json, camera_id)
        D = self.read_distortion_vector(calibration_json, camera_id)
        rectification = self.read_custom_json(rectification_json_path)
        R = np.array(rectification["R"][camera_id])
        P = np.array(rectification["P"][camera_id])
        # rectify maps
        stereoMap_x, stereoMap_y = cv2.initUndistortRectifyMap(K, D, R, P, (self.width,self.height),
                                                                    cv2.CV_16SC2)
        return stereoMap_x,stereoMap_y

    # @jit(parallel = True,fastmath = True)
    @jit(device=True)
    def rectify(self,image,stereoMap_x, stereoMap_y):
        # executing maps
        img_rect = cv2.remap(image, stereoMap_x, stereoMap_y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
        return img_rect


    def left_image_callback(self, msg):
        left_image =self.bridge.imgmsg_to_cv2(msg, "mono8")
        # cv2.imwrite("/home/avena/nvidia_ws/src/basler_ros2_driver/test/left_raw.png",left_image)
        start = time.time()
        left_image_rect = self.rectify(left_image, self.stereoMapL_x,self.stereoMapL_y)
        end = time.time()
        print((end - start)*1000)
        # cv2.imwrite("/home/avena/nvidia_ws/src/basler_ros2_driver/test/left_rect.png",left_image_rect)
        left_image_rect_msg = self.bridge.cv2_to_imgmsg(left_image_rect, "mono8")
        left_image_rect_msg.header =  msg.header
        self.left_rect_publisher.publish(left_image_rect_msg)

    def right_image_callback(self, msg):
        right_image =self.bridge.imgmsg_to_cv2(msg, "mono8")
        # cv2.imwrite("/home/avena/nvidia_ws/src/basler_ros2_driver/test/right_raw.png",right_image)
        right_image_rect = self.rectify(right_image, self.stereoMapR_x,self.stereoMapR_y)
        # cv2.imwrite("/home/avena/nvidia_ws/src/basler_ros2_driver/test/right_rect.png",right_image_rect)
        right_image_rect_msg = self.bridge.cv2_to_imgmsg(right_image_rect, "mono8")
        right_image_rect_msg.header = msg.header
        self.right_rect_publisher.publish(right_image_rect_msg)

    

def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()