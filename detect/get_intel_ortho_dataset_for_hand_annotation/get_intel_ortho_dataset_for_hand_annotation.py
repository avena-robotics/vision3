from typing import Tuple
import argparse
import time
import numpy as np
import os
import cv2
import open3d as o3d
from numba import jit
import sys

# NOTE: This path is assuming that system was setup  
# according to documentation
sys.path.append(os.path.join(os.path.expanduser('~'), 'vision3'))
from intelcam.open_intel_camera import open_intel_camera
from intelcam.close_intel_camera import close_intel_camera
from intelcam.get_intel_all_images import get_intel_all_images


@jit(parallel=True, fastmath=True)
def calculate_rgbd_orthophoto(points: np.ndarray, colors: np.ndarray, voxel_size: float = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates orthographic view from colored point cloud
    
    This function is responsible for calculating orthographic view using colored point cloud.
    It first voxelizes provided 3D points in XY plane with specified voxel size.  

    Args:
        points: array with 3D coordinates of point cloud (each coordinate is assumed to be in millimeter) (np.ndarray),
        colors: array with RGB pixel (each channel in pixel is assumed to be in range [0, 1] (np.ndarray),
        voxel_size: size of each voxel in XY plane in millimeters (float); default: 1, 

    Returns:
        Tuple with 2 elements:
            - rgb_array: array with RGB pixels (each channel in range [0, 255] in orthographic view (np.ndarray),
            - depth_array: array with depth values (in millimeters) in orthographic view (np.ndarray)
    """
    # Convert [0; 1] RGB to [0; 255] and cast to uint8
    colors = (colors * 255).astype(np.uint8)

    # Convert type to int32
    points = points.astype(np.int32)
    
    # Voxelize in XY plane
    points[:, 0] = (points[:, 0] - points[:, 0].min()) / voxel_size
    points[:, 1] = (points[:, 1] - points[:, 1].min()) / voxel_size
    
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


# Filters
def trunc_depth(depth_frame, d):
    """Truncate depth image.
    
    This function is responsible for truncating (zero out)
    pixels which values are greater then specified minimal distance.

    Args:
        depth_frame: image with depth values (in millimeters),
        d: maximum distance allowed (in millimeters)

    Returns:
        depth image with values greater than d set to 0
    """
    arr_img = depth_frame.copy()
    arr_img[arr_img >= d] = 0
    return arr_img


def bilateral_filter(depth_frame, d, sigma_color, sigma_space):
    """Applies bilateral filter to input image.
    
    This function is responsible for filtering input image
    by applying bilateral filtration which is highly effective
    in noise removal while keeping edges sharp. This function
    is to be used for depth images filtration.

    Args:
        depth_frame: input depth frame in UINT16 format with millimeters,
        d: dimeter of each pixel's neighborhood,
        sigma_color: filter sigma in the color space,
        sigma_space: filter sigma in the coordinate space,
    
    Returns:
        filtered depth image in UINT16 bit depth in millimeters
    """
    arr_img = depth_frame.astype(np.float32)
    arr_img = cv2.bilateralFilter(arr_img, d, sigma_color, sigma_space)
    arr_img = arr_img.astype(np.uint16)
    return arr_img


def statistical_outlier_removal(point_cloud, nb_neighbors, std_ratio):
    """Filters Open3D point cloud with statistical outlier removal. 
    
    This function is responsible for filtering input point cloud
    by applying statistical analysis for each point in the cloud.

    Args:
        point_cloud: input cloud to be filtered tensor version (open3d.t.geometry.PointCloud),
        nb_neighbors: number of closest points used to determine whether
                      point is an outlier or inlier,
        std_ratio: standard deviation ratio.

    Returns:
        filtered point cloud (open3d.t.geometry.PointCloud)
    """
    point_cloud = point_cloud.to_legacy()
    _, ind = point_cloud.remove_statistical_outlier(nb_neighbors, std_ratio)
    point_cloud = point_cloud.select_by_index(ind)
    point_cloud_t = o3d.t.geometry.PointCloud.from_legacy_pointcloud(point_cloud)
    return point_cloud_t


def median_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Median blur filtration
    
    This function is responsible for applying median filter on input
    image using points from square neighborhood of size kernel_size.

    NOTE: When kernel_size is 3 or 5, the image bit depth should be CV_8U, 
    CV_16U, or CV_32F, for larger kernel sizes, it can only be CV_8U.

    Args:
        image: input image to be filtered (numpy.ndarray),
        kernel_size: size of square neighborhood to be used for filtration

    Returns:
        filtered image with the size and dtype as the input image.
    """
    array_median = cv2.medianBlur(image, kernel_size)
    return array_median


def create_point_cloud(color: np.ndarray, depth: np.ndarray, camera_info: dict):
    """Convert RGB and depth image to Open3D point cloud using camera intrinsic.

    This function is responsible for creating Open3D point cloud using aligned 
    color and depth image with camera intrinsic.

    Args:
        color: ndarray with color RGB image, dtype is UINT8 (numpy.ndarray),
        depth: ndarray with depth value in millimeters, dtype is UINT16 (numpy.ndarray)
        camera_info: intrinsic parameters for color images (i.e fx, fy, cx, cy, width, height)

    Returns:
        created point cloud (open3d.geometry.PointCloud)
    """
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color=o3d.geometry.Image(color),
                                                                    depth=o3d.geometry.Image(depth),
                                                                    depth_trunc=10000, # depth is filtered before in trunc_depth
                                                                    depth_scale=1,
                                                                    convert_rgb_to_intensity=False)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width=camera_info["width"],
                                                  height=camera_info["height"],
                                                  fx=camera_info["fx"], fy=camera_info["fy"],
                                                  cx=camera_info["cx"], cy=camera_info["cy"])
    pcld = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_image,
                                                          intrinsic=intrinsic)
    return pcld


if __name__ == '__main__':
    WIDTH = 1000
    HEIGHT = 720

    #########################################################
    # Getting command line arguments from user
    # dir_default_path = os.path.dirname(os.path.abspath(__file__))
    config_default_path = os.path.join(os.path.expanduser("~"), 'vision3', 'intelcam', 'config', 'cam9_config.json')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--base_dir', type=str, 
                         help='absolute path to directory', nargs='?',
                         default=f'{os.path.join(os.path.expanduser("~"), "dataset")}')
    parser.add_argument('-c', '--configuration', type=str,
                        help='path to camera configuration file', nargs='?',
                        default=f'{config_default_path}')
    parser.add_argument('-s', '--voxel_size', type=float,
                        help='voxel grid size in millimeters', nargs='?',
                        default=1)
    args = parser.parse_args()
    base_dir = args.base_dir
    intel_configuration_file = args.configuration
    voxel_size = args.voxel_size
    #########################################################

    print(f'Saving images to "{base_dir}" directory')
    print(f'Reading camera configuration file: "{intel_configuration_file}"')
    print(f'Set voxel size is: {voxel_size}')

    # Creating directory for images
    os.makedirs(base_dir, exist_ok=True)

    # Open Intel camera
    intel_camera_handle, camera_info = open_intel_camera(intel_configuration_file)
    print('Opened cameras successfully')
    try:
        pressed_key = input('Pressed ENTER to make a photo (or "q" and then ENTER to exit): ')
        cnt = 1
        while pressed_key != 'q':
            print('Reading color and depth images')
            try:
                color, depth = get_intel_all_images(intel_camera_handle)
            except:
                # Probably timeout occured while reading images.
                # Set arrays to empty ones
                color, depth = np.array([]), np.array([])          

            if color.size == 0 or depth.size == 0:
                raise RuntimeError('Failed to get images. Reconnect cameras and run script again')

            # NOTE: Zero out pixels which are out of view (value read empirically)
            cutoff_val = 238
            color[:, :cutoff_val, :] = [0, 0, 0]
            depth[:, :cutoff_val] = 0

            print('Filtering depth in Z axis')
            depth = trunc_depth(depth, 1500)

            # Create point cloud using Open3D
            print('Creating point cloud from RGBD image')
            pcld = create_point_cloud(color, depth, camera_info)

            # Get copy of points and colors
            points = np.asarray(pcld.points)
            colors = np.asarray(pcld.colors)

            # Calculate orthographic view
            print('Calculating orthographic view')
            rgb_array, depth_array = calculate_rgbd_orthophoto(points, colors, voxel_size)

            # Crop ortho view images
            w_min = 0
            w_max = w_min + WIDTH
            h_min = 0
            h_max = h_min + HEIGHT
            rgb_array = rgb_array[h_min:h_max, w_min:w_max, :]
            depth_array = depth_array[h_min:h_max, w_min:w_max]

            # Make sure that images are WIDTHxHEIGHT
            # Sometimes because of noise, number of rows might
            # be less than desired so here we are adding black lines
            # so result resolution is always the same
            rows_add = HEIGHT - rgb_array.shape[0]
            color_rows = np.zeros((rows_add, rgb_array.shape[1], 3), dtype=rgb_array.dtype)
            rgb_array = np.vstack((rgb_array, color_rows))
            depth_rows = np.zeros((rows_add, depth_array.shape[1]), dtype=depth_array.dtype)
            depth_array = np.vstack((depth_array, depth_rows))
            
            # Save images
            ts_now = time.time_ns()
            print(f'Saving images to "{base_dir}" directory')
            cv2.imwrite(os.path.join(base_dir, f'{ts_now}_depth.png'), depth_array)
            cv2.imwrite(os.path.join(base_dir, f'{ts_now}_color.png'), cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR))
            print(f'Save images called {cnt} times')
            cnt += 1
            pressed_key = input('Pressed ENTER to make a photo (or "q" and then ENTER to exit): ')
    except Exception as e:
        print(f'[ERROR]: {e}')
    
    close_intel_camera(intel_camera_handle)
