from cv_bridge import CvBridge
from custom_interfaces.srv import GetMonoImages
import numpy as np
from basler_client_functions.call_empty_service_and_get_data import call_empty_service_and_get_data


def get_basler_mono_images() -> tuple:
    """
    This function is responsible to call ROS2 service which is responsible for 
    getting two mono images from cameras.

    :return tuple(np.ndarray, np.ndarray) with left mono and right mono (in that order) or empty arrays if it fails
    """
    service_result: GetMonoImages.Response = call_empty_service_and_get_data('get_basler_mono_images', GetMonoImages)
    if service_result:
        if service_result.left_mono.width == 0 or service_result.right_mono.width == 0:
            return np.array([]), np.array([])
        bridge = CvBridge()
        left_mono = bridge.imgmsg_to_cv2(service_result.left_mono)                
        right_mono = bridge.imgmsg_to_cv2(service_result.right_mono) 
        return left_mono, right_mono
    return np.array([]), np.array([])
