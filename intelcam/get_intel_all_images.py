from typing import Tuple
import numpy as np
import pyrealsense2 as rs


def get_intel_all_images(camera_handle) -> Tuple[np.ndarray, np.ndarray]:
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
    # Aligner for depth to color
    align_to = rs.stream.color
    aligner = rs.align(align_to)
    # Wait for synchronized color and depth frames
    frames = camera_handle.wait_for_frames(timeout_ms=2000)
    # Align depth to color frame
    aligned_frames = aligner.process(frames)
    # Depth
    depth_frame = aligned_frames.get_depth_frame()
    # Temporal filter
    temporal_filter = rs.temporal_filter()
    filter_smooth_alpha = rs.option.filter_smooth_alpha
    filter_smooth_delta = rs.option.filter_smooth_delta
    temporal_smooth_alpha = 0.03
    temporal_smooth_delta = 60
    temporal_filter.set_option(filter_smooth_alpha, temporal_smooth_alpha)
    temporal_filter.set_option(filter_smooth_delta, temporal_smooth_delta)
    # Spatial filter
    spatial_filter = rs.spatial_filter()
    filter_magnitude = rs.option.filter_magnitude
    spatial_magnitude = 2
    spatial_smooth_alpha = 0.5
    spatial_smooth_delta = 1
    spatial_filter.set_option(filter_smooth_alpha, spatial_smooth_alpha)
    spatial_filter.set_option(filter_smooth_delta, spatial_smooth_delta)
    spatial_filter.set_option(filter_magnitude, spatial_magnitude)
    # Process depth with filters
    depth_frame = temporal_filter.process(depth_frame)
    # depth_frame = spatial_filter.process(depth_frame)
    depth_image = np.asanyarray(depth_frame.get_data())
    # Color
    color_frame = aligned_frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    return color_image, depth_image
