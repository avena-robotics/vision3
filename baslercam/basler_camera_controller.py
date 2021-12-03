# basler_camera_controller.py / Main ros node
import cv2
from datetime import datetime

from pypylon import genicam
from pypylon import pylon

from custom_interfaces.srv import GetAllImages, GetMonoImages, GetColorImage

from basler_drive import BaslerPyDriver

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.executors import ExternalShutdownException
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

def main(args=None):
    rclpy.init(args=args)
    # define yout serials
    camera_group_serials = {'left_mono': '40132642', 'color': '40120189', 'right_mono': '40134745'}
    # define ros2 node class for offering services
    basler_py_driver_node = BaslerPyDriver(qos_profile_sensor_data,camera_group_serials)
    try:
        while rclpy.ok():
            rclpy.spin_once(basler_py_driver_node)
            if basler_py_driver_node._avena_basler_cameras is not None and basler_py_driver_node._avena_basler_cameras.IsOpen() and not basler_py_driver_node._avena_basler_cameras.IsGrabbing():
                basler_py_driver_node.get_logger().warn("cameras are not grabbing")
    except KeyboardInterrupt:
        basler_py_driver_node.get_logger().warn("exit key is pressed")
        if basler_py_driver_node._avena_basler_cameras is not None and basler_py_driver_node._avena_basler_cameras.GetSize() > 0:
            basler_py_driver_node.close_basler_cameras(basler_py_driver_node._avena_basler_cameras)
    except ExternalShutdownException:
        basler_py_driver_node.get_logger().warn("external shutdown")
        if basler_py_driver_node._avena_basler_cameras is not None and basler_py_driver_node._avena_basler_cameras.GetSize() > 0:
            basler_py_driver_node.close_basler_cameras(basler_py_driver_node._avena_basler_cameras)
    finally:
        rclpy.try_shutdown()
        basler_py_driver_node.destroy_node()

if __name__ == '__main__':
    main()
