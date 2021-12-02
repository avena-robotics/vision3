from std_srvs.srv import Trigger
from basler_client_functions.call_empty_service_and_get_data import call_empty_service_and_get_data


def close_basler_cameras() -> bool:
    """
    This function is responsible to call ROS2 service which is responsible for 
    closing Basler cameras.
    :return bool whether call to service succeeded (True -> success, False -> failure).
    """
    service_result: Trigger.Response = call_empty_service_and_get_data('close_basler_cameras', Trigger)
    return False if service_result is None else service_result.success
