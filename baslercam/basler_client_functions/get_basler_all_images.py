from cv_bridge import CvBridge
from custom_interfaces.srv import GetAllImages
import numpy as np
from basler_client_functions.call_empty_service_and_get_data import call_empty_service_and_get_data


def get_basler_all_images() -> tuple:
    """
    This function is responsible to call ROS2 service which is responsible for 
    getting two mono and color images from cameras.

    :return tuple(np.ndarray, np.ndarray, np.ndarray) with left mono, color and right mono (in that order) or (np.array([]), np.array([]), np.array([])) if it fails
    """
    service_result: GetAllImages.Response = call_empty_service_and_get_data('get_basler_all_images', GetAllImages)
    if service_result:
        if service_result.left_mono.width == 0 or service_result.right_mono.width == 0 or service_result.color.width == 0:
            return np.array([]), np.array([]), np.array([])
        bridge = CvBridge()
        left_mono = bridge.imgmsg_to_cv2(service_result.left_mono)                
        right_mono = bridge.imgmsg_to_cv2(service_result.right_mono) 
        color = bridge.imgmsg_to_cv2(service_result.color)  
        return left_mono, color, right_mono
    return np.array([]), np.array([]), np.array([])
