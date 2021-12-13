from typing import Tuple
import open3d as o3d
import json
import time
import numpy as np
import os
# import cupyx
# from cupyx.scipy import ndimage
import copy
import cv2
from numba import jit


def open_intel_camera(config_filename):
    """
    TODO
    """
    # Load JSON with configuration
    with open(config_filename) as cf:
        rs_cfg = o3d.t.io.RealSenseSensorConfig(json.load(cf))

    # Initialize device and get metadata
    rscam = o3d.t.io.RealSenseSensor()
    rscam.init_sensor(rs_cfg)
    rgbd_metadata = rscam.get_metadata()

    # Get intrinsic parameters from metadata
    fx = rgbd_metadata.intrinsics.intrinsic_matrix[0][0]
    fy = rgbd_metadata.intrinsics.intrinsic_matrix[1][1]
    ppx = rgbd_metadata.intrinsics.intrinsic_matrix[0][2]
    ppy = rgbd_metadata.intrinsics.intrinsic_matrix[1][2]
    height = rgbd_metadata.height
    width = rgbd_metadata.width

    # Start acquiring images
    rscam.start_capture(start_record=False)
    print('Waiting 2 seconds for camera to start...\n')
    time.sleep(1)

    # Return results
    return rscam, {"fx": fx, "fy": fy, "cx": ppx, "cy": ppy, "height": height, "width": width}


def get_intel_all_images(camera_handle) -> Tuple[np.ndarray, np.ndarray]:
    """
    TODO
    """
    rgbd_frame = camera_handle.capture_frame(align_depth_to_color=True)
    color = rgbd_frame.color
    depth = rgbd_frame.depth
    return color, depth


def close_intel_camera(camera_handle):
    """
    TODO
    """
    camera_handle.stop_capture()


@jit(parallel=True, fastmath=True)
def calculate_rgbd_orthophoto(points: np.ndarray, colors: np.ndarray, voxel_size: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates orthographic view from colored point cloud
    
    This function is responsible for calculating orthographic view using colored point cloud.
    It first voxelizes provided 3D points in XY plane with specified voxel size.  

    Args:
        points: array with 3D coordinates of point cloud (each coordinate is assumed to be in meters) (np.ndarray),
        colors: array with RGB pixel (each channel in pixel is assumed to be in range [0, 1] (np.ndarray),
        voxel_size: size of each voxel in XY plane in meters (float); default: 0.001, 

    Returns:
        Tuple with 2 elements:
            - rgb_array: array with RGB pixels (each channel in range [0, 255] in orthographic view (np.ndarray),
            - depth_array: array with depth values (in millimeters) in orthographic view (np.ndarray)
    """
    # Convert [0; 1] RGB to [0; 255] and cast to uint8
    colors = (colors * 255).astype(np.uint8)

    # Voxelize in XY plane
    points[:, 0] = (points[:, 0] - points[:, 0].min()) / voxel_size
    points[:, 1] = (points[:, 1] - points[:, 1].min()) / voxel_size
    
    # Convert meters to millimeters in Z axis
    points[:, 2] = points[:, 2] * 1000
    
    # Cast to UINT16 type
    points = points.astype(np.uint16)

    # Get bounding box
    min_x = points[:, 0].min()
    max_x = points[:, 0].max() + 1
    min_y = points[:, 1].min()
    max_y = points[:, 1].max() + 1
    min_z = points[:, 2].min()
    max_z = points[:, 2].max() + 1

    # Prepare output arrays for depth and RGB
    depth_array = np.full((int(max_y - min_y), int(max_x - min_x)), max_z, dtype=np.uint16)
    rgb_array = np.zeros((int(max_y - min_y), int(max_x - min_x), 3), dtype=np.uint8)   
    
    for i in range(points.shape[0]):
        z = points[i, 2]
        x = points[i, 1]
        y = points[i, 0]

        if z < depth_array[x, y]:
            depth_array[x, y] = z
            rgb_array[x, y] = colors[i, :]
            
    # Zero out all invalid pixels
    invalid_mask = (depth_array == max_z)
    depth_array[invalid_mask] = 0
    
    return rgb_array, depth_array

# # Utility function to convert RGB and depth in orthographic view to Open3D point cloud
# def create_point_cloud_from_orthophoto_view(rgb: np.ndarray, depth: np.ndarray) -> o3d.geometry.PointCloud:
#     points = []
#     colors = []

#     for r in range(depth.shape[0]):
#         for c in range(depth.shape[1]):
#             if depth[r, c] != 0:
#                 points.append([r, c, depth[r, c]])
#                 colors.append(rgb[r, c] / 255)

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(np.array(points))
#     pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
#     return pcd


# Filters
def trunc_depth(depth_frame, d):
    """
    TODO
    """
    arr_img = np.asarray(depth_frame.to_legacy_image())
    arr_img[arr_img >= d] = 0
    tensor = o3d.core.Tensor(arr_img)
    image = o3d.t.geometry.Image(tensor)
    return image


def bilateral_filter(depth_frame, d, sigma_color, sigma_space):
    """
    TODO
    """
    arr_img = np.asarray(depth_frame.to_legacy_image()).astype(np.float32)
    arr_filtered = cv2.bilateralFilter(arr_img, d, sigma_color, sigma_space)
    tensor = o3d.core.Tensor(arr_filtered.astype(np.uint16))
    image = o3d.t.geometry.Image(tensor)
    return image


def statistical_outlier_removal(point_cloud, nb_neighbors, std_ratio):
    """
    TODO
    """
    point_cloud = point_cloud.to_legacy_pointcloud()
    _, ind = point_cloud.remove_statistical_outlier(nb_neighbors, std_ratio)
    point_cloud = point_cloud.select_by_index(ind)
    point_cloud_gpu = o3d.t.geometry.PointCloud.from_legacy_pointcloud(point_cloud)
    return point_cloud_gpu


def median_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Median blur
    TODO
    """
    depth_array_median = cv2.medianBlur(image, kernel_size)
    return depth_array_median


if __name__ == '__main__':
    #########################################################
    # User modifies path for dataset
    BASE_DIR = '/home/avena/software/intel/dataset001'
    intel_configuration_file = '/home/avena/vision/vision/intel/input/cam9_config.json'
    voxel_size = 0.001
    median_filter_kernel = 5
    #########################################################

    # Creating directory for images
    os.makedirs(BASE_DIR, exist_ok=True)

    # Open Intel camera
    intel_camera_handle, camera_info = open_intel_camera(intel_configuration_file)
    print('Opened cameras successfully')
    try:
        pressed_key = input('Pressed ENTER to make a photo (or "q" and then ENTER to exit): ')
        cnt = 1
        while pressed_key != 'q':
            print('Reading color and depth images')
            color, depth = get_intel_all_images(intel_camera_handle)
            if color.columns == 0 or depth.columns == 0:
                print('Failed to get images. Try to reopen camera...')
                close_intel_camera(intel_camera_handle)
                intel_camera_handle, camera_info = open_intel_camera(intel_configuration_file)
                continue
            
            print('Filtering depth in Z axis')
            depth = trunc_depth(depth, 2000)

            # Create point cloud using Open3D
            print('Creating point cloud from RGBD image')
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color=color.to_legacy_image(),
                                                                        depth=depth.to_legacy_image(),
                                                                        depth_trunc=1.2,
                                                                        convert_rgb_to_intensity=False)
            intrinsic = o3d.camera.PinholeCameraIntrinsic(width=camera_info["width"],
                                                    height=camera_info["height"],
                                                    fx=camera_info["fx"], fy=camera_info["fy"],
                                                    cx=camera_info["cx"], cy=camera_info["cy"])
            pcld = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_image,
                                                            intrinsic=intrinsic)

            print('Calculating orthographic view')
            # Get copy of points and colors
            points = np.asarray(copy.deepcopy(pcld.points))
            colors = np.asarray(copy.deepcopy(pcld.colors))

            # Calculate orthographic view
            rgb_array, depth_array = calculate_rgbd_orthophoto(points, colors, voxel_size)

            print('Filtering')
            # Orthographic view RGB and depth filtration
            # Median filter
            # rgb_array_median = cv2.medianBlur(rgb_array, 7)
            rgb_array_median = median_blur(rgb_array, median_filter_kernel)
            depth_array_median = median_blur(depth_array, median_filter_kernel)

            # Save images
            ts_now = time.time_ns()
            print(f'Saving images to "{BASE_DIR}" directory')
            cv2.imwrite(os.path.join(BASE_DIR, f'{ts_now}_depth.png'), depth_array_median)
            cv2.imwrite(os.path.join(BASE_DIR, f'{ts_now}_color.png'), cv2.cvtColor(rgb_array_median, cv2.COLOR_RGB2BGR))
            print(f'Save images called {cnt} times')
            cnt += 1
            pressed_key = input('Pressed ENTER to make a photo (or "q" and then ENTER to exit): ')
    except Exception as e:
        print(f'[ERROR]: {e}')
        pass
    
    close_intel_camera(intel_camera_handle)
