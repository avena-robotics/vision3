import numpy as np
from cv_bridge import CvBridge
from custom_interfaces.srv import GetColorImage
from basler_client_functions.call_empty_service_and_get_data import call_empty_service_and_get_data


def get_basler_color_image() -> np.ndarray:
    """
    This function is responsible to call ROS2 service which is responsible for 
    getting single color image from cameras
    :return np.ndarray with color image data or empty array if it fails
    """
    service_result: GetColorImage.Response = call_empty_service_and_get_data('get_basler_color_image', GetColorImage)
    if service_result:
        bridge = CvBridge()
        color_image = bridge.imgmsg_to_cv2(service_result.color)
        return color_image
    return np.array([])
