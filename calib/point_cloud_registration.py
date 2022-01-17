# Required packages
import time
import open3d as o3d
import numpy as np
import sys
import os
import copy
import cv2
import json
import argparse
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.expanduser('~'), 'vision3'))
from intelcam.close_intel_camera import close_intel_camera
from intelcam.open_intel_camera import open_intel_camera
from intelcam.get_intel_all_images import get_intel_all_images
np.set_printoptions(suppress=True)


def tsdf_scene_cloud(frames_color,
                     frames_depth,
                     camera_info,
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

    extrinsic_gpu = o3d.core.Tensor(np.linalg.inv(np.eye(4)), o3d.core.Dtype.Float32, device)
    pinhole_temporary = o3d.camera.PinholeCameraIntrinsic(width=camera_info["width"],
                                                          height=camera_info["height"],
                                                          fx=camera_info["fx"], fy=camera_info["fy"],
                                                          cx=camera_info["cx"], cy=camera_info["cy"])
    intrinsic_gpu = o3d.core.Tensor(pinhole_temporary.intrinsic_matrix, o3d.core.Dtype.Float32, device)

    for color, depth in zip(frames_color, frames_depth):
        # print(type(color))
        # print(type(depth))
        color_gpu = color.to(device)
        depth_gpu = depth.to(device)
        volume.integrate(depth=depth_gpu,
                         color=color_gpu,
                         intrinsics=intrinsic_gpu,
                         extrinsics=extrinsic_gpu,
                         depth_scale=1000.0,
                         depth_max=1.2,
                         )

    # Measuring times
    point_cloud = volume.cuda(0).extract_surface_points()
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
         target_temp.to_legacy()],)
        # zoom=0.5,
        # front=[-0.2458, -0.8088, 0.5342],
        # lookat=[1.7745, 2.2305, 0.9787],
        # up=[0.3109, -0.5878, -0.7468])


def tensor_colored_icp(source: o3d.t.geometry.PointCloud, target: o3d.t.geometry.PointCloud):
    # Fit plane and extract
    print('Removing table points from point clouds')
    source_legacy = source.to_legacy()
    _, inliers = source_legacy.segment_plane(distance_threshold=0.01,
                                            ransac_n=3,
                                            num_iterations=1000)
    outlier_cloud = source_legacy.select_by_index(inliers, invert=True)
    source = o3d.t.geometry.PointCloud.from_legacy(outlier_cloud)

    target_legacy = target.to_legacy()
    _, inliers = target_legacy.segment_plane(distance_threshold=0.01,
                                            ransac_n=3,
                                            num_iterations=1000)
    outlier_cloud = target_legacy.select_by_index(inliers, invert=True)
    target = o3d.t.geometry.PointCloud.from_legacy(outlier_cloud)
    o3d.visualization.draw([source, target])

    # init_tf = np.array([[-0.99997, -0.00399, -0.01043, -0.01052],
    #                                   [0.01065, -0.53904, -0.84164, 0.91812],
    #                                   [-0.00297, -0.84169, 0.53908, 0.50068],
    #                                   [0.00000, 0.00000, 0.00000, 1.00000]])
    # source = source.clone().transform(init_tf)

    print('Running ICP')
    source_cp = source.clone()
    target_cp = target.clone()
    source_cp.estimate_normals()
    target_cp.estimate_normals()
    source_cp.point["colors"] = source_cp.point["colors"].to(o3d.core.Dtype.Float32) / 255.0
    target_cp.point["colors"] = target_cp.point["colors"].to(o3d.core.Dtype.Float32) / 255.0

    init_source_to_target = np.identity(4)
    estimation = o3d.t.pipelines.registration.TransformationEstimationForColoredICP()
    criteria_list = [
        o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.0001,
                                                            relative_rmse=0.0001,
                                                            max_iteration=69),
        o3d.t.pipelines.registration.ICPConvergenceCriteria(0.00001, 0.00001, 30),
        o3d.t.pipelines.registration.ICPConvergenceCriteria(0.000001, 0.000001, 14)
    ]
    max_correspondence_distances = o3d.utility.DoubleVector([0.08, 0.04, 0.02])
    voxel_sizes = o3d.utility.DoubleVector([0.04, 0.02, 0.01])
    reg_multiscale_icp = o3d.t.pipelines.registration.multi_scale_icp(source_cp, target_cp, voxel_sizes,
                                                                      criteria_list,
                                                                      max_correspondence_distances,
                                                                      init_source_to_target, estimation)

    print("Fitness:", reg_multiscale_icp.fitness)
    print("Inlier RMSE:", reg_multiscale_icp.inlier_rmse)
    print("Estimated transformation:\n", reg_multiscale_icp.transformation)
    draw_registration_result(source, target, reg_multiscale_icp.transformation)
    # draw_registration_result(source, target, np.linalg.inv(init_source_to_target))
    # draw_registration_result(source, target, init_source_to_target)


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, 
                                                                    max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return pcd_down, pcd_fpfh


def global_point_clouds_registration(source: o3d.t.geometry.PointCloud, target: o3d.t.geometry.PointCloud) -> np.ndarray:
    source_legacy = source.to_legacy()
    target_legacy = target.to_legacy()
    voxel_size = 0.01
    source_down, source_fpfh = preprocess_point_cloud(source_legacy, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_legacy, voxel_size)
    # o3d.visualization.draw([source_down, target_down])

    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled.")
    print("   Since the downsampling voxel size is", voxel_size)
    print("   we use a liberal distance threshold", distance_threshold)
    # Global
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3, 
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], 
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    print(result_ransac)
    return result_ransac.transformation


def local_point_clouds_registration(source: o3d.t.geometry.PointCloud, target: o3d.t.geometry.PointCloud, initial_tf: np.ndarray) -> np.ndarray:
    source.estimate_normals()
    target.estimate_normals()
    criteria_list = [
        o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.0001,
                                                            relative_rmse=0.0001,
                                                            max_iteration=50*10),
        o3d.t.pipelines.registration.ICPConvergenceCriteria(0.00001, 0.00001, 30*10),
        o3d.t.pipelines.registration.ICPConvergenceCriteria(0.000001, 0.000001, 14*10)
    ]
    # voxel_sizes = o3d.utility.DoubleVector([0.01, 0.005, 0.002])
    # max_correspondence_distances = o3d.utility.DoubleVector([0.02, 0.01, 0.004])

    max_correspondence_distances = o3d.utility.DoubleVector([0.02, 0.01, 0.004])
    voxel_sizes = o3d.utility.DoubleVector([0.01, 0.005, 0.002])

    result_icp = o3d.t.pipelines.registration.multi_scale_icp(source=source, 
                                                              target=target, 
                                                              voxel_sizes=voxel_sizes,
                                                              criteria_list=criteria_list,
                                                              max_correspondence_distances=max_correspondence_distances,
                                                              init_source_to_target=initial_tf, 
                                                              estimation_method=o3d.t.pipelines.registration.TransformationEstimationForColoredICP(),
                                                              save_loss_log=True,
    )
    # result_icp.save_loss_log = True
    print('fitness:', result_icp.fitness, ', inlier_rmse:', result_icp.inlier_rmse)
    # print(result_icp.loss_log)

    return result_icp.transformation


def remove_plane(point_cloud: o3d.t.geometry.PointCloud) -> o3d.t.geometry.PointCloud:
    target_legacy = point_cloud.to_legacy()
    _, inliers = target_legacy.segment_plane(distance_threshold=0.01,
                                        ransac_n=3,
                                        num_iterations=1000)
    outlier_cloud = target_legacy.select_by_index(inliers, invert=True)
    return o3d.t.geometry.PointCloud.from_legacy(outlier_cloud)


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('cam0', help='id of the target camera')
        parser.add_argument('cam1', help='id of the source camera')
        parser.add_argument('-n', '--nr_frames', help='number of frames used for TSDF', default=50, type=int)
        args = parser.parse_args()
        target, source = args.cam0, args.cam1
        nr_frames = args.nr_frames
        device = o3d.core.Device("CUDA:0")
        
        initial_tf = np.array(json.load(open(f'charuco_calibration_cam{source}_to_cam{target}.json', 'r')))
        cam0_handle, cam0_info = open_intel_camera(f'/home/avena/vision3/intelcam/config/cam{source}_config.json',)
        cam1_handle, cam1_info = open_intel_camera(f'/home/avena/vision3/intelcam/config/cam{target}_config.json',)

        # Cam0
        # cam0_mask = cv2.imread(f'cam{source}_mask.png', cv2.IMREAD_GRAYSCALE)
        color_frames, depth_frames = [], []
        for _ in range(nr_frames):
            color, depth = get_intel_all_images(cam0_handle)
            # color = cv2.copyTo(color, cam0_mask)
            # depth = cv2.copyTo(depth, cam0_mask)
            # color[:, :color.shape[1] // 2, :] = [0, 0, 0]
            # depth[:, :depth.shape[1] // 2] = 0
            color_frames.append(o3d.t.geometry.Image.from_legacy(o3d.geometry.Image(color)))
            depth_frames.append(o3d.t.geometry.Image.from_legacy(o3d.geometry.Image(depth)))
        cam0_point_cloud = tsdf_scene_cloud(color_frames, depth_frames, cam0_info, device)

        # Cam1
        color_frames, depth_frames = [], []
        # cam1_mask = cv2.imread(f'cam{target}_mask.png', cv2.IMREAD_GRAYSCALE)
        for _ in range(nr_frames):
            color, depth = get_intel_all_images(cam1_handle)
            # color = cv2.copyTo(color, cam1_mask)
            # depth = cv2.copyTo(depth, cam1_mask)
            # color[:, :color.shape[1] // 2, :] = [0, 0, 0]
            # depth[:, :depth.shape[1] // 2] = 0
            color_frames.append(o3d.t.geometry.Image.from_legacy(o3d.geometry.Image(color)))
            depth_frames.append(o3d.t.geometry.Image.from_legacy(o3d.geometry.Image(depth)))
        
        cam1_point_cloud = tsdf_scene_cloud(color_frames, depth_frames, cam1_info, device)

        # initial_tf = global_point_clouds_registration(cam0_point_cloud, cam1_point_cloud)
        print('Initial transform:')
        print(repr(initial_tf))
        print('Global calibration result')
        draw_registration_result(cam0_point_cloud, cam1_point_cloud, initial_tf)
        
        pressed_key = input('Continue with local registration? (y/n): ')
        if pressed_key.lower() == 'y':        
            tf = local_point_clouds_registration(cam0_point_cloud, cam1_point_cloud, initial_tf)
            calib_file_path = f'local_calibration_cam{args.cam1}_to_cam{args.cam0}.json'
            print(f'Saving calibration to "{calib_file_path}"')
            with open(calib_file_path, 'w') as f:
                json.dump(tf.numpy().tolist(), f)

            print('After calibration:')
            print(repr(tf.numpy()))
            print('Local registration result')
            draw_registration_result(cam0_point_cloud, cam1_point_cloud, tf)

    except Exception as e:
        print('[ERROR]:', e)

    close_intel_camera(cam0_handle)
    close_intel_camera(cam1_handle)


# # Required packages
# import time
# import open3d as o3d
# import numpy as np
# import sys
# import os
# import copy
# import cv2
# import json
# import argparse
# import matplotlib.pyplot as plt
# sys.path.append(os.path.join(os.path.expanduser('~'), 'vision3'))
# from intelcam.close_intel_camera import close_intel_camera
# from intelcam.open_intel_camera import open_intel_camera
# from intelcam.get_intel_all_images import get_intel_all_images
# np.set_printoptions(suppress=True)


# def tsdf_scene_cloud(frames_color,
#                      frames_depth,
#                      camera_info,
#                      device):
#     volume = o3d.t.geometry.TSDFVoxelGrid(
#         map_attrs_to_dtypes={
#             'tsdf': o3d.core.Dtype.Float32,
#             'weight': o3d.core.Dtype.UInt16,
#             'color': o3d.core.Dtype.UInt16
#         },
#         voxel_size=0.001,
#         sdf_trunc=0.005,
#         block_resolution=16,
#         block_count=50000,
#         device=device, )

#     extrinsic_gpu = o3d.core.Tensor(np.linalg.inv(np.eye(4)), o3d.core.Dtype.Float32, device)
#     pinhole_temporary = o3d.camera.PinholeCameraIntrinsic(width=camera_info["width"],
#                                                           height=camera_info["height"],
#                                                           fx=camera_info["fx"], fy=camera_info["fy"],
#                                                           cx=camera_info["cx"], cy=camera_info["cy"])
#     intrinsic_gpu = o3d.core.Tensor(pinhole_temporary.intrinsic_matrix, o3d.core.Dtype.Float32, device)

#     for color, depth in zip(frames_color, frames_depth):
#         # print(type(color))
#         # print(type(depth))
#         color_gpu = color.to(device)
#         depth_gpu = depth.to(device)
#         volume.integrate(depth=depth_gpu,
#                          color=color_gpu,
#                          intrinsics=intrinsic_gpu,
#                          extrinsics=extrinsic_gpu,
#                          depth_scale=1000.0,
#                          depth_max=1.2,
#                          )

#     # Measuring times
#     point_cloud = volume.cuda(0).extract_surface_points()
#     o3d.core.cuda.release_cache()
#     return point_cloud


# def tsdf_scene_cloud_multiple_cameras(frames_color,
#                      frames_depth,
#                      cameras_info,
#                      transforms,
#                      device):
#     volume = o3d.t.geometry.TSDFVoxelGrid(
#         map_attrs_to_dtypes={
#             'tsdf': o3d.core.Dtype.Float32,
#             'weight': o3d.core.Dtype.UInt16,
#             'color': o3d.core.Dtype.UInt16
#         },
#         voxel_size=0.001,
#         sdf_trunc=0.005,
#         block_resolution=16,
#         block_count=50000,
#         device=device, )

#     extrinsic_gpu = []
#     intrinsic_gpu = []

#     for camera_info, transform in zip(cameras_info, transforms):
#         # extrinsic_gpu.append(o3d.core.Tensor(transform, o3d.core.Dtype.Float32, device))
#         extrinsic_gpu.append(o3d.core.Tensor(np.linalg.inv(transform), o3d.core.Dtype.Float32, device))
#         pinhole_temporary = o3d.camera.PinholeCameraIntrinsic(width=camera_info["width"],
#                                                               height=camera_info["height"],
#                                                               fx=camera_info["fx"], fy=camera_info["fy"],
#                                                               cx=camera_info["cx"], cy=camera_info["cy"])
#         intrinsic_gpu.append(o3d.core.Tensor(pinhole_temporary.intrinsic_matrix, o3d.core.Dtype.Float32, device))

#     pure_tsdf_start = time.time()
#     for i in range(len(frames_depth)):
#         for j in range(len(frames_depth[i])):
#             color_gpu = frames_color[i][j].to(device)
#             depth_gpu = frames_depth[i][j].to(device)
#             volume.integrate(depth=depth_gpu,
#                              color=color_gpu,
#                              intrinsics=intrinsic_gpu[i],
#                              extrinsics=extrinsic_gpu[i],
#                              depth_scale=1000.0,
#                              depth_max=1.2,
#                              )

#     # Measuring times
#     pure_tsdf_no_extract_end = time.time()
#     print(f"Without extracting point cloud from mesh it took {pure_tsdf_no_extract_end - pure_tsdf_start} seconds.")
#     point_cloud = volume.cuda(0).extract_surface_points()
#     pure_tsdf_end = time.time()
#     print(f"TSDF has finished its job. It took {pure_tsdf_end - pure_tsdf_start} seconds.")
#     o3d.core.cuda.release_cache()
#     return point_cloud


# def draw_registration_result(source: o3d.t.geometry.PointCloud, target: o3d.t.geometry.PointCloud, transformation):
#     source_temp = source.clone()
#     target_temp = target.clone()

#     source_temp.transform(transformation)

#     # This is patched version for tutorial rendering.
#     # Use `draw` function for you application.
#     o3d.visualization.draw(
#         [source_temp.to_legacy(),
#          target_temp.to_legacy()],)
#         # zoom=0.5,
#         # front=[-0.2458, -0.8088, 0.5342],
#         # lookat=[1.7745, 2.2305, 0.9787],
#         # up=[0.3109, -0.5878, -0.7468])


# def tensor_colored_icp(source: o3d.t.geometry.PointCloud, target: o3d.t.geometry.PointCloud):
#     # Fit plane and extract
#     print('Removing table points from point clouds')
#     source_legacy = source.to_legacy()
#     _, inliers = source_legacy.segment_plane(distance_threshold=0.01,
#                                             ransac_n=3,
#                                             num_iterations=1000)
#     outlier_cloud = source_legacy.select_by_index(inliers, invert=True)
#     source = o3d.t.geometry.PointCloud.from_legacy(outlier_cloud)

#     target_legacy = target.to_legacy()
#     _, inliers = target_legacy.segment_plane(distance_threshold=0.01,
#                                             ransac_n=3,
#                                             num_iterations=1000)
#     outlier_cloud = target_legacy.select_by_index(inliers, invert=True)
#     target = o3d.t.geometry.PointCloud.from_legacy(outlier_cloud)
#     o3d.visualization.draw([source, target])

#     # init_tf = np.array([[-0.99997, -0.00399, -0.01043, -0.01052],
#     #                                   [0.01065, -0.53904, -0.84164, 0.91812],
#     #                                   [-0.00297, -0.84169, 0.53908, 0.50068],
#     #                                   [0.00000, 0.00000, 0.00000, 1.00000]])
#     # source = source.clone().transform(init_tf)

#     print('Running ICP')
#     source_cp = source.clone()
#     target_cp = target.clone()
#     source_cp.estimate_normals()
#     target_cp.estimate_normals()
#     source_cp.point["colors"] = source_cp.point["colors"].to(o3d.core.Dtype.Float32) / 255.0
#     target_cp.point["colors"] = target_cp.point["colors"].to(o3d.core.Dtype.Float32) / 255.0

#     init_source_to_target = np.identity(4)
#     estimation = o3d.t.pipelines.registration.TransformationEstimationForColoredICP()
#     criteria_list = [
#         o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.0001,
#                                                             relative_rmse=0.0001,
#                                                             max_iteration=69),
#         o3d.t.pipelines.registration.ICPConvergenceCriteria(0.00001, 0.00001, 30),
#         o3d.t.pipelines.registration.ICPConvergenceCriteria(0.000001, 0.000001, 14)
#     ]
#     max_correspondence_distances = o3d.utility.DoubleVector([0.08, 0.04, 0.02])
#     voxel_sizes = o3d.utility.DoubleVector([0.04, 0.02, 0.01])
#     reg_multiscale_icp = o3d.t.pipelines.registration.multi_scale_icp(source_cp, target_cp, voxel_sizes,
#                                                                       criteria_list,
#                                                                       max_correspondence_distances,
#                                                                       init_source_to_target, estimation)

#     print("Fitness:", reg_multiscale_icp.fitness)
#     print("Inlier RMSE:", reg_multiscale_icp.inlier_rmse)
#     print("Estimated transformation:\n", reg_multiscale_icp.transformation)
#     draw_registration_result(source, target, reg_multiscale_icp.transformation)
#     # draw_registration_result(source, target, np.linalg.inv(init_source_to_target))
#     # draw_registration_result(source, target, init_source_to_target)


# def preprocess_point_cloud(pcd, voxel_size):
#     print(":: Downsample with a voxel size %.3f." % voxel_size)
#     pcd_down = pcd.voxel_down_sample(voxel_size)

#     radius_normal = voxel_size * 2
#     print(":: Estimate normal with search radius %.3f." % radius_normal)
#     pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, 
#                                                                     max_nn=30))

#     radius_feature = voxel_size * 5
#     print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
#     pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#         pcd_down,
#         o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
#     )
#     return pcd_down, pcd_fpfh


# def global_point_clouds_registration(source: o3d.t.geometry.PointCloud, target: o3d.t.geometry.PointCloud) -> np.ndarray:
#     source_legacy = source.to_legacy()
#     target_legacy = target.to_legacy()
#     voxel_size = 0.01
#     source_down, source_fpfh = preprocess_point_cloud(source_legacy, voxel_size)
#     target_down, target_fpfh = preprocess_point_cloud(target_legacy, voxel_size)
#     # o3d.visualization.draw([source_down, target_down])

#     distance_threshold = voxel_size * 1.5
#     print(":: RANSAC registration on downsampled.")
#     print("   Since the downsampling voxel size is", voxel_size)
#     print("   we use a liberal distance threshold", distance_threshold)
#     # Global
#     result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
#         source_down, target_down, source_fpfh, target_fpfh, True,
#         distance_threshold,
#         estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
#         ransac_n=3, 
#         checkers=[
#             o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
#             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
#         ], 
#         criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
#     )
#     print(result_ransac)
#     return result_ransac.transformation


# def local_point_clouds_registration(source: o3d.t.geometry.PointCloud, target: o3d.t.geometry.PointCloud, initial_tf: np.ndarray) -> np.ndarray:
#     source.estimate_normals()
#     target.estimate_normals()
#     criteria_list = [
#         o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.0001,
#                                                             relative_rmse=0.0001,
#                                                             max_iteration=50*10),
#         o3d.t.pipelines.registration.ICPConvergenceCriteria(0.00001, 0.00001, 30*10),
#         o3d.t.pipelines.registration.ICPConvergenceCriteria(0.000001, 0.000001, 14*10)
#     ]
#     # voxel_sizes = o3d.utility.DoubleVector([0.01, 0.005, 0.002])
#     # max_correspondence_distances = o3d.utility.DoubleVector([0.02, 0.01, 0.004])

#     max_correspondence_distances = o3d.utility.DoubleVector([0.02, 0.01, 0.004])
#     voxel_sizes = o3d.utility.DoubleVector([0.01, 0.005, 0.002])

#     result_icp = o3d.t.pipelines.registration.multi_scale_icp(source=source, 
#                                                               target=target, 
#                                                               voxel_sizes=voxel_sizes,
#                                                               criteria_list=criteria_list,
#                                                               max_correspondence_distances=max_correspondence_distances,
#                                                               init_source_to_target=initial_tf, 
#                                                               estimation_method=o3d.t.pipelines.registration.TransformationEstimationForColoredICP(),
#                                                               save_loss_log=True,
#     )
#     # result_icp.save_loss_log = True
#     print('fitness:', result_icp.fitness, ', inlier_rmse:', result_icp.inlier_rmse)
#     # print(result_icp.loss_log)

#     return result_icp.transformation


# def remove_plane(point_cloud: o3d.t.geometry.PointCloud) -> o3d.t.geometry.PointCloud:
#     target_legacy = point_cloud.to_legacy()
#     _, inliers = target_legacy.segment_plane(distance_threshold=0.01,
#                                         ransac_n=3,
#                                         num_iterations=1000)
#     outlier_cloud = target_legacy.select_by_index(inliers, invert=True)
#     return o3d.t.geometry.PointCloud.from_legacy(outlier_cloud)


# if __name__ == '__main__':
#     try:
#         parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#         # parser.add_argument('cam0', help='id of the target camera')
#         # parser.add_argument('cam1', help='id of the source camera')
#         parser.add_argument('-n', '--nr_frames', help='number of frames used for TSDF', default=50, type=int)
#         args = parser.parse_args()
#         # target, source = args.cam0, args.cam1
#         nr_frames = args.nr_frames
#         device = o3d.core.Device("CUDA:0")
        
#         tf_cam0_to_cam2 = np.array(json.load(open(f'local_calibration_cam0_to_cam2.json', 'r')))
#         initial_tf_cam2_to_cam3 = np.array(json.load(open(f'charuco_calibration_cam2_to_cam3.json', 'r')))

#         cam0_handle, cam0_info = open_intel_camera(f'/home/avena/vision3/intelcam/config/cam0_config.json')
#         cam2_handle, cam2_info = open_intel_camera(f'/home/avena/vision3/intelcam/config/cam2_config.json')
#         cam3_handle, cam3_info = open_intel_camera(f'/home/avena/vision3/intelcam/config/cam3_config.json')

#         # Cam0
#         cam0_mask = cv2.imread(f'cam0_mask.png', cv2.IMREAD_GRAYSCALE)
#         cam0_color_frames, cam0_depth_frames = [], []
#         for _ in range(nr_frames):
#             color, depth = get_intel_all_images(cam0_handle)
#             color = cv2.copyTo(color, cam0_mask)
#             depth = cv2.copyTo(depth, cam0_mask)
#             cam0_color_frames.append(o3d.t.geometry.Image.from_legacy(o3d.geometry.Image(color)))
#             cam0_depth_frames.append(o3d.t.geometry.Image.from_legacy(o3d.geometry.Image(depth)))

#         # Cam2
#         cam2_mask = cv2.imread(f'cam2_mask.png', cv2.IMREAD_GRAYSCALE)
#         cam2_color_frames, cam2_depth_frames = [], []
#         for _ in range(nr_frames):
#             color, depth = get_intel_all_images(cam2_handle)
#             color = cv2.copyTo(color, cam2_mask)
#             depth = cv2.copyTo(depth, cam2_mask)
#             cam2_color_frames.append(o3d.t.geometry.Image.from_legacy(o3d.geometry.Image(color)))
#             cam2_depth_frames.append(o3d.t.geometry.Image.from_legacy(o3d.geometry.Image(depth)))        
#         cam2_point_cloud = tsdf_scene_cloud_multiple_cameras([cam2_color_frames, cam0_color_frames], 
#                                                         [cam2_depth_frames, cam0_depth_frames], 
#                                                         [cam2_info,         cam0_info], 
#                                                         [np.identity(4),    tf_cam0_to_cam2],
#                                                         device)
#         # Cam3
#         cam3_mask = cv2.imread(f'cam3_mask.png', cv2.IMREAD_GRAYSCALE)
#         color_frames, depth_frames = [], []
#         for _ in range(nr_frames):
#             color, depth = get_intel_all_images(cam3_handle)
#             color = cv2.copyTo(color, cam3_mask)
#             depth = cv2.copyTo(depth, cam3_mask)
#             color_frames.append(o3d.t.geometry.Image.from_legacy(o3d.geometry.Image(color)))
#             depth_frames.append(o3d.t.geometry.Image.from_legacy(o3d.geometry.Image(depth)))
#         cam3_point_cloud = tsdf_scene_cloud(color_frames, depth_frames, cam3_info, device)

#         # cam2_point_cloud = cam2_point_cloud.transform(initial_tf_cam2_to_cam3)
#         # o3d.visualization.draw([cam2_point_cloud, cam3_point_cloud])

#         # initial_tf = global_point_clouds_registration(cam0_point_cloud, cam1_point_cloud)
#         # print('Initial transform:')
#         # print(repr(initial_tf))
#         print('Global calibration result')
#         draw_registration_result(cam2_point_cloud, cam3_point_cloud, initial_tf_cam2_to_cam3)
        
#         pressed_key = input('Continue with local registration? (y/n): ')
#         if pressed_key.lower() == 'y':        
#             tf = local_point_clouds_registration(cam2_point_cloud, cam3_point_cloud, initial_tf_cam2_to_cam3)
#             calib_file_path = f'local_calibration_cam2_to_cam3.json'
#             print(f'Saving calibration to "{calib_file_path}"')
#             with open(calib_file_path, 'w') as f:
#                 json.dump(tf.numpy().tolist(), f)

#             print('After calibration:')
#             print(repr(tf.numpy()))
#             print('Local registration result')
#             draw_registration_result(cam2_point_cloud, cam3_point_cloud, tf)

#     except Exception as e:
#         print('[ERROR]:', e)

#     close_intel_camera(cam0_handle)
#     close_intel_camera(cam2_handle)
#     close_intel_camera(cam3_handle)
