# Required packages
import time
from cv2 import polarToCart
import open3d as o3d
import numpy as np
import sys
import os
import copy
import cv2
import json
from numba import jit
from typing import Tuple
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.expanduser('~'), 'vision3'))
from intelcam.close_intel_camera import close_intel_camera
from intelcam.open_intel_camera import open_intel_camera
from intelcam.get_intel_all_images import get_intel_all_images
from image_processing.utils import colormap_image


def tsdf_scene_cloud(frames_color,
                     frames_depth,
                     cameras_info,
                     transforms,
                     device):
    volume = o3d.t.geometry.TSDFVoxelGrid(
        map_attrs_to_dtypes={
            'tsdf': o3d.core.Dtype.Float32,
            'weight': o3d.core.Dtype.UInt16,
            'color': o3d.core.Dtype.UInt16
        },
        voxel_size=0.001,
        sdf_trunc=0.005,
        block_resolution=16,
        block_count=50000,
        device=device, )

    extrinsic_gpu = []
    intrinsic_gpu = []

    for camera_info, transform in zip(cameras_info, transforms):
        # extrinsic_gpu.append(o3d.core.Tensor(transform, o3d.core.Dtype.Float32, device))
        extrinsic_gpu.append(o3d.core.Tensor(np.linalg.inv(transform), o3d.core.Dtype.Float32, device))
        pinhole_temporary = o3d.camera.PinholeCameraIntrinsic(width=camera_info["width"],
                                                              height=camera_info["height"],
                                                              fx=camera_info["fx"], fy=camera_info["fy"],
                                                              cx=camera_info["cx"], cy=camera_info["cy"])
        intrinsic_gpu.append(o3d.core.Tensor(pinhole_temporary.intrinsic_matrix, o3d.core.Dtype.Float32, device))

    pure_tsdf_start = time.time()
    for i in range(len(frames_depth)):
        for j in range(len(frames_depth[i])):
            color_gpu = frames_color[i][j].to(device)
            depth_gpu = frames_depth[i][j].to(device)
            volume.integrate(depth=depth_gpu,
                             color=color_gpu,
                             intrinsics=intrinsic_gpu[i],
                             extrinsics=extrinsic_gpu[i],
                             depth_scale=1000.0,
                             depth_max=1.2,
                             )

    # Measuring times
    pure_tsdf_no_extract_end = time.time()
    print(f"Without extracting point cloud from mesh it took {pure_tsdf_no_extract_end - pure_tsdf_start} seconds.")
    point_cloud = volume.cuda(0).extract_surface_points()
    pure_tsdf_end = time.time()
    print(f"TSDF has finished its job. It took {pure_tsdf_end - pure_tsdf_start} seconds.")
    o3d.core.cuda.release_cache()
    return point_cloud


def draw_registration_result(source: o3d.t.geometry.PointCloud, target: o3d.t.geometry.PointCloud, transformation):
    source_temp = source.clone()
    target_temp = target.clone()

    source_temp.transform(transformation)

    # This is patched version for tutorial rendering.
    # Use `draw` function for you application.
    o3d.visualization.draw(
        [source_temp.to_legacy(),
         target_temp.to_legacy()])


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


def get_rgbd_images(camera_handle, nr_frames: int):
    cam_color_frames, cam_depth_frames = [], []
    for _ in range(nr_frames):
        color, depth = get_intel_all_images(camera_handle)
        cam_color_frames.append(o3d.t.geometry.Image.from_legacy(o3d.geometry.Image(color)))
        cam_depth_frames.append(o3d.t.geometry.Image.from_legacy(o3d.geometry.Image(depth)))
    return cam_color_frames, cam_depth_frames


if __name__ == '__main__':
    try:
        nr_frames = 8
        device = o3d.core.Device("CUDA:0")

        cam0_handle, cam0_info = open_intel_camera('/home/avena/vision3/intelcam/config/cam0_config.json')
        cam1_handle, cam1_info = open_intel_camera('/home/avena/vision3/intelcam/config/cam1_config.json')
        cam2_handle, cam2_info = open_intel_camera('/home/avena/vision3/intelcam/config/cam2_config.json')
        cam3_handle, cam3_info = open_intel_camera('/home/avena/vision3/intelcam/config/cam3_config.json')
        cam4_handle, cam4_info = open_intel_camera('/home/avena/vision3/intelcam/config/cam4_config.json')
        cam5_handle, cam5_info = open_intel_camera('/home/avena/vision3/intelcam/config/cam5_config.json')

        # Read cameras images
        cam0_color_frames, cam0_depth_frames = get_rgbd_images(cam0_handle, nr_frames)
        cam1_color_frames, cam1_depth_frames = get_rgbd_images(cam1_handle, nr_frames)
        cam2_color_frames, cam2_depth_frames = get_rgbd_images(cam2_handle, nr_frames)
        cam3_color_frames, cam3_depth_frames = get_rgbd_images(cam3_handle, nr_frames)
        cam4_color_frames, cam4_depth_frames = get_rgbd_images(cam4_handle, nr_frames)
        cam5_color_frames, cam5_depth_frames = get_rgbd_images(cam5_handle, nr_frames)

        # Transform
        tf_0_to_4 = np.array(json.load(open("calib/local_calibration_cam0_to_cam4.json")))
        tf_1_to_4 = np.array(json.load(open("calib/local_calibration_cam1_to_cam4.json")))
        tf_2_to_4 = np.array(json.load(open("calib/local_calibration_cam2_to_cam4.json")))
        tf_3_to_4 = np.array(json.load(open("calib/local_calibration_cam3_to_cam4.json")))
        tf_5_to_4 = np.array(json.load(open("calib/local_calibration_cam5_to_cam4.json")))

        # tf_0_to_4 = np.array(json.load(open("calib/charuco_calibration_cam0_to_cam4.json")))
        # tf_1_to_4 = np.array(json.load(open("calib/charuco_calibration_cam1_to_cam4.json")))
        # tf_2_to_4 = np.array(json.load(open("calib/charuco_calibration_cam2_to_cam4.json")))
        # tf_3_to_4 = np.array(json.load(open("calib/charuco_calibration_cam3_to_cam4.json")))
        # tf_5_to_4 = np.array(json.load(open("calib/charuco_calibration_cam5_to_cam4.json")))
        
        scene_point_cloud = tsdf_scene_cloud([cam4_color_frames, cam0_color_frames, cam1_color_frames, cam2_color_frames, cam3_color_frames, cam5_color_frames],
                                             [cam4_depth_frames, cam0_depth_frames, cam1_depth_frames, cam2_depth_frames, cam3_depth_frames, cam5_depth_frames],
                                             [cam4_info,         cam0_info,         cam1_info,         cam2_info,         cam3_info,         cam5_info],
                                             [np.identity(4), tf_0_to_4, tf_1_to_4, tf_2_to_4, tf_3_to_4, tf_5_to_4],
                                             device)

        # scene_point_cloud = tsdf_scene_cloud([ cam0_color_frames,  cam2_color_frames, ],
        #                                      [ cam0_depth_frames,  cam2_depth_frames, ],
        #                                      [ cam0_info,          cam2_info,         ],
        #                                      [ tf_0_to_4,          tf_2_to_4,         ],
        #                                      device)

        # scene_point_cloud0 = tsdf_scene_cloud([cam0_color_frames, ],
        #                                      [cam0_depth_frames, ],
        #                                      [cam0_info,         ],
        #                                      [tf_0_to_4],
        #                                      device)
        # scene_point_cloud1 = tsdf_scene_cloud([cam1_color_frames, ],
        #                                      [cam1_depth_frames, ],
        #                                      [cam1_info,         ],
        #                                      [tf_1_to_4],
        #                                      device)
        # scene_point_cloud2 = tsdf_scene_cloud([cam2_color_frames, ],
        #                                      [cam2_depth_frames, ],
        #                                      [cam2_info,         ],
        #                                      [tf_2_to_4],
        #                                      device)
        # scene_point_cloud3 = tsdf_scene_cloud([cam3_color_frames, ],
        #                                      [cam3_depth_frames, ],
        #                                      [cam3_info,         ],
        #                                      [tf_3_to_4],
        #                                      device)
        # scene_point_cloud = scene_point_cloud0 + scene_point_cloud1 + scene_point_cloud2 + scene_point_cloud3

        # tf_to_table = np.array(json.load(open('calib/calibration_cam4_to_table.json')))

        # scene_point_cloud = scene_point_cloud.transform(np.linalg.inv(tf_to_table))

        #############################################################
        o3d.visualization.draw(scene_point_cloud)
        ts = time.time_ns()
        o3d.io.write_point_cloud(f'{ts}_scene.ply', scene_point_cloud.to_legacy())
        #############################################################

        # Get copy of points and colors
        scene_point_cloud = scene_point_cloud.cpu()
        points = np.asanyarray(scene_point_cloud.to_legacy().points) * 1000
        colors = np.asanyarray(scene_point_cloud.to_legacy().colors)

        # Calculate orthographic view
        print('Calculating orthographic view')
        rgb_array, depth_array = calculate_rgbd_orthophoto(points, colors, 1)
        
        #############################################################
        plt.figure()
        plt.imshow(rgb_array)
        plt.figure()
        plt.imshow(depth_array)
        plt.show()

        cv2.imwrite(f'{ts}_color.png', cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'{ts}_depth_colormap.png', colormap_image(depth_array))
        cv2.imwrite(f'{ts}_depth.png', depth_array)
        #############################################################
        
        # o3d.io.write_point_cloud('three_side_intel_d415_tsdf_scene_cloud.ply', scene_point_cloud.to_legacy())
        # draw_registration_result(cam0_point_cloud, cam1_point_cloud, tf)

    except Exception as e:
        print('[ERROR]:', e)

    close_intel_camera(cam0_handle)
    # close_intel_camera(cam1_handle)
    close_intel_camera(cam2_handle)
