import cv2
import time
import numpy as np

import calc_rectification
import calc_disparity
import calc_depth


##############################################################################
# Configuration
first_camera_name = 'left_mono'
second_camera_name = 'right_mono'
calib_config_path = '/home/avena/vision3/calib/basler_calibration.json'
left_image_path = f'/home/avena/left_mono.png'
right_image_path = f'/home/avena/right_mono.png'
max_disparity = 512

out_depth = '/home/avena/vision3/3d_processing/depth.png'
##############################################################################

##############################################################################
# Initialization
# Rectification initialization
# Extracting rectification params from read JSON
left_cam_mat, left_dist_coeffs, right_cam_mat, right_dist_coeffs, left_P, right_P, left_R, right_R, left_imsize, right_imsize = calc_rectification.get_rectification_params(calib_config_path, first_camera_name, second_camera_name)

# Initialize remap maps
left_stereo_map_x, left_stereo_map_y = calc_rectification.init_rect_maps(left_cam_mat, left_dist_coeffs, 
                                                        left_R, left_P, left_imsize)
right_stereo_map_x, right_stereo_map_y = calc_rectification.init_rect_maps(right_cam_mat, right_dist_coeffs, 
                                                        right_R, right_P, right_imsize)
# Depth
# Reading Q matrix (perspective transformation matrix from calibration results)
q = calc_depth.get_perspective_transform(calib_config_path, first_camera_name, second_camera_name)
##############################################################################


##############################################################################
# Reading images
print('Reading images...')
left_image = cv2.imread(left_image_path)
right_image = cv2.imread(right_image_path)
print('...done')
##############################################################################


##############################################################################
# Rectification
print('Rectification calculation...')
start = time.perf_counter()
left_image_rect = calc_rectification.calc_rectification(left_image, left_stereo_map_x, left_stereo_map_y)
print(f'Rectification time (one image): {(time.perf_counter() - start) * 1000} [ms]')
right_image_rect = calc_rectification.calc_rectification(right_image, right_stereo_map_x, right_stereo_map_y)
print('...done')
##############################################################################


##############################################################################
# Disparity
print('Disparity calculation...')
start = time.perf_counter()
disparity = calc_disparity.calc_disparity(left_image_rect, right_image_rect, max_disparity=max_disparity)
# print(disparity)
print(disparity.min())
print(disparity.max())
print(disparity.dtype)
print(disparity.shape)

# Saving disparity to file
# print('###')
disp_cp = disparity.copy()
disp_cp[disp_cp == disp_cp.min()] = 0
# print(disp_cp)
# print(disp_cp.min())
# print(disp_cp.max())
# print(disp_cp.dtype)
# print(disp_cp.shape)
cv2.imwrite('disparity.png', (disp_cp / disp_cp.max() * 255).astype(np.uint8))
print(f'Calculating disparity time: {(time.perf_counter() - start) * 1000} [ms]')
print('...done')
##############################################################################


##############################################################################
# Depth
# print('Depth calculation...')
depth = calc_depth.calc_depth(disparity, q)

print('table:\n', depth[360:365, 1130:1153])
print('box:\n', depth[1950:1955, 970:975])

# # print(depth)
# print(depth.min())
# print(depth.max())
# # print(depth.dtype)
# # print(depth.shape)
# # print(image_3d.shape)

# # calc_depth.save_point_cloud_to_obj('cloud.obj', image_3d)

depth_cp = depth.copy()
depth_cp[depth_cp == depth_cp.min()] = 0
cv2.imwrite('depth.png', (depth_cp / depth_cp.max() * 255).astype(np.uint8))
print('...done')
##############################################################################
