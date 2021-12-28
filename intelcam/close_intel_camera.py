def close_intel_camera(camera_handle):
    """Closes Intel Realsense camera.
    
    This function is responsible for closing camera 
    to stop streaming data.

    Args:
        camera_handle: handler for camera which is used also to get images
    """
    camera_handle.stop()
