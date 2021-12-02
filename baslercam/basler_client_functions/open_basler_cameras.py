from basler_client_functions.call_empty_service_and_get_data import call_empty_service_and_get_data
from std_srvs.srv import Trigger


def open_basler_cameras() -> bool:
    """
    This function is responsible to call ROS2 service which is responsible for 
    opening Basler cameras.
    :return bool whether cameras were opened successfully (True -> success, False -> failure)
    """
    service_result: Trigger.Response = call_empty_service_and_get_data('open_basler_cameras', Trigger)
    return False if service_result is None else service_result.success
