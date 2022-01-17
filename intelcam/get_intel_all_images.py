from typing import Tuple
import numpy as np
import pyrealsense2 as rs
import cv2

def get_intel_all_images(camera_handle, num_of_frames: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Reads last from Intel camera.
    
    This function is responsible for reading last color and depth 
    frame in which depth is aligned to color frame. Color is RGB 
    where each channel is in range [0; 255] and depth is distance
    in millimeters.

    Args:
        camera_handle: handle to Intel camera (it is assumed that camera is opened)

    Returns:
        Tuple with 2 elements:
            - color array: each channel is in [0; 255] range (np.ndarray),
            - depth array: distance in millimeters (np.ndarray)
    """
    # Temporal filter
    temporal_filter = rs.temporal_filter()
    filter_smooth_alpha = rs.option.filter_smooth_alpha
    filter_smooth_delta = rs.option.filter_smooth_delta
    # temporal_smooth_alpha = 0.01
    # temporal_smooth_delta = 20
    temporal_smooth_alpha = 0.5
    temporal_smooth_delta = 20
    temporal_filter.set_option(filter_smooth_alpha, temporal_smooth_alpha)
    temporal_filter.set_option(filter_smooth_delta, temporal_smooth_delta)

    # Spatial filter
    spatial_filter = rs.spatial_filter()
    filter_magnitude = rs.option.filter_magnitude
    spatial_magnitude = 2
    spatial_smooth_alpha = 0.6
    spatial_smooth_delta = 8
    spatial_filter.set_option(filter_smooth_alpha, spatial_smooth_alpha)
    spatial_filter.set_option(filter_smooth_delta, spatial_smooth_delta)
    spatial_filter.set_option(filter_magnitude, spatial_magnitude)

    # Aligner for depth to color
    align_to = rs.stream.color
    aligner = rs.align(align_to)

    # It is recommended to filter disparity and then convert back to depth:
    # https://dev.intelrealsense.com/docs/depth-post-processing
    # Disparity to depth transformer
    disparity_to_depth = rs.disparity_transform(False)

    # Depth to disparity transformer
    depth_to_disparity = rs.disparity_transform(True)

    for _ in range(num_of_frames):
        # Wait for synchronized color and depth frames
        frames = camera_handle.wait_for_frames(timeout_ms=2000)
        # Align depth to color frame
        aligned_frames = aligner.process(frames)
        # Depth
        depth_frame = aligned_frames.get_depth_frame()
        # Convert to disparity
        disparity_frame = depth_to_disparity.process(depth_frame)
        # Process depth with filters
        disparity_frame = temporal_filter.process(disparity_frame)
        disparity_frame = spatial_filter.process(disparity_frame)
    
    # # Convert disparity to depth
    # disparity_filter = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
    # disparity_filter.setDepthDiscontinuityRadius(3)
    # disparity_filter.setLambda(8000)
    # disparity_filter.setLRCthresh(24)
    # disparity_filter.setSigmaColor(1)
    # # disparity_filter.setROI((0, 0, 0, 0))
    
    # Color
    color_frame = aligned_frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())

    print(help(type(disparity_frame.get_data())))

    # disparity_frame = np.ascontiguousarray(disparity_frame.get_data(), dtype=np.float32)
    # print(disparity_frame.dtype)
    # print(disparity_frame.shape)
    # print(np.min(disparity_frame))
    # print(np.max(disparity_frame))

    # import matplotlib.pyplot as plt
    # plt.imshow(disparity_frame)
    # plt.show()
    # disparity_frame = disparity_filter.filter(disparity_frame, color_frame)

    depth_frame = disparity_to_depth.process(disparity_frame)

    # Depth
    depth_image = np.asanyarray(depth_frame.get_data())

    return color_image, depth_image
