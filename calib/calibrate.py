#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary packages
import glob
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import json
from scipy.spatial.transform import Rotation as R


# In[2]:


# Configuration
path_to_folder = '/home/avena/dataset001'
output_file = 'basler_calibration.json'
acceptance_threshold = 60  # minimal number of detected markers 
                           # on each image when calibrating in stereo

# Aruco dictionary definition 
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)

# Charuco board definition
board = cv2.aruco.CharucoBoard_create(squaresX=12, 
                                      squaresY=9, 
                                      squareLength=0.06,  # meters 
                                      markerLength=0.047, # meters
                                      dictionary=aruco_dict)


# In[3]:


# Execute script
def analyze_charuco(images, aruco_dict, board):
    """Detect Charuco corners.
    
    This function is responsible for detecting Charuco markers corners
    on each image with subpixel accuracy. 
    This function is used by functions which calculate intrinsic parameters
    or calibrate cameras in stereo setup.
        
    Args:
        images: list of path to images
        aruco_dict: object with information about used Aruco markers
        board: object with full description of used Charuco board with real sizes
        
    Returns:
        Tuple with 3 elements:
            - all_corners
            - all_ids
            - imsize
    """
        
    all_corners = []
    all_ids = []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.00001)
    for img_path in images:
        # Read image as grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
        # Detect ChAruCo corners and refine found points
        marker_corners, ids, rejected_img_points = cv2.aruco.detectMarkers(
                                                                img, 
                                                                aruco_dict)
        marker_corners, ids, refusd, recoverd = cv2.aruco.refineDetectedMarkers(
                                                            img, board, 
                                                            marker_corners, ids, 
                                                            rejected_img_points)

        if len(marker_corners) > 0:
            # Improve accuracy of found corners and store them in lists
            num_corners, charuco_corners, charuco_ids =                         cv2.aruco.interpolateCornersCharuco(
                                                    marker_corners, ids, 
                                                    img, board)
            if num_corners > 3:
                charuco_corners = cv2.cornerSubPix(image=img, 
                                                   corners=charuco_corners,
                                                   winSize=(5, 5),
                                                   zeroZone=(-1, -1),
                                                   criteria=criteria)
                all_corners.append(charuco_corners) 
                all_ids.append(charuco_ids)           
    imsize = img.shape[::-1]
    return all_corners, all_ids, imsize


def calibrate_camera(all_corners, all_ids, imsize):
    """Calibrate camera intrinsic parameters.
    
    This function is responsible for calibrating camera 
    i.e. calculating intrinsic parameters from detected marker corners.
        
    Args:
        all_corners: detected Charuco corners coordinates
        all_ids: detected Charuco corners unique IDs
        imsize: resolution of image on which markers were detected
        
    Returns:
        Tuple with 3 elements:
            - reprojection_error
            - camera_matrix
            - distortion_coefficients
    """
    # Initialize camera matrix:
    # [[fx,  0, cx],
    #  [ 0, fy, cy],
    #  [ 0,  0,  1]]
    fx_init = np.max(imsize) / 2
    camera_matrix_init = np.array([[fx_init,     0.0, imsize[0] / 2],
                                   [    0.0, fx_init, imsize[1] / 2],
                                   [    0.0,     0.0,           1.0]])
    dist_coeffs_init = np.zeros((5, 1))
    flags = cv2.CALIB_USE_INTRINSIC_GUESS
    (reprojection_error, camera_matrix, distortion_coefficients,
     rotation_vectors, translation_vectors,
     std_deviations_intrinsics, std_deviations_extrinsics,
     per_view_errors) = cv2.aruco.calibrateCameraCharucoExtended(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=board,
        imageSize=imsize,
        cameraMatrix=camera_matrix_init,
        distCoeffs=dist_coeffs_init,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

    return reprojection_error, camera_matrix, distortion_coefficients


def calibrate_stereo_camera(all_corners_l, all_ids_l, all_corners_r, all_ids_r, imsize, 
                            camera_matrix_l, dist_coeff_l, camera_matrix_r, dist_coeff_r, 
                            board, acceptance_threshold=108):
    """Calibrate cameras in stereo setup.
    
    This function is responsible for calibrating cameras in stereo setup i.e. finding
    transform (translation and rotation) between left and right camera in setup.
    
    Args:
        all_corners_l: detected corners on left camera, 
        all_ids_l: : detected corners IDs on left camera, 
        all_corners_r: : detected corners in right camera, 
        all_ids_r: detected corners IDs in right camera, 
        imsize: resolution of image, 
        camera_matrix_l: left camera matrix,
        dist_coeff_l: distortion model coefficients for left camera, 
        camera_matrix_r: right camera matrix,
        dist_coeff_r: distortion model coefficients for right camera, 
        board: object with description of real Charuco board, 
        acceptance_threshold: minimal number of markers that have to 
                              detected on images from both cameras (default is 108 = 9 x 12 markers)
        
    Returns:
        Tuple with 3 elements:
            - reprojection error,
            - rotation between left and right camera,
            - translation between left and right camera.
    """
    left_corners_sampled = []
    right_corners_sampled = []
    obj_pts = []
    one_pts = board.chessboardCorners
    for i in range(len(all_ids_l)):
        left_sub_corners = []
        right_sub_corners = []
        obj_pts_sub = []
        
        if len(all_ids_l[i]) < acceptance_threshold or            len(all_ids_r[i]) < acceptance_threshold:
            continue
        for j in range(len(all_ids_l[i])):
            idx = np.where(all_ids_r[i] == all_ids_l[i][j])
            if idx[0].size == 0:
                continue
            left_sub_corners.append(all_corners_l[i][j])
            right_sub_corners.append(all_corners_r[i][idx])
            obj_pts_sub.append(one_pts[all_ids_l[i][j]])

        obj_pts.append(np.array(obj_pts_sub, dtype=np.float32))
        left_corners_sampled.append(np.array(left_sub_corners, dtype=np.float32))
        right_corners_sampled.append(np.array(right_sub_corners, dtype=np.float32))

    stereocalib_criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

    flags = cv2.CALIB_FIX_INTRINSIC
    
    if len(left_corners_sampled) == 0 or len(right_corners_sampled) == 0:
        raise RuntimeError('Not enough common samples when calibrating stereo. '
                           'Try decreasing "acceptance_threshold" parameter')
    
    ret_val = cv2.stereoCalibrate(
        obj_pts, left_corners_sampled, right_corners_sampled,
        camera_matrix_l, dist_coeff_l, camera_matrix_r, dist_coeff_r, imsize,
        criteria=stereocalib_criteria, flags=flags)
    
    reprojection_error = ret_val[0]
    rotation = ret_val[5]
    translation = ret_val[6]
    
    return reprojection_error, rotation, translation


def calc_rectification_maps(left_cam_mat, left_dist_coeffs, 
                            right_cam_mat, right_dist_coeffs, 
                            rot_matrix, trans_vec,
                            imsize):
    """Calculate rectification parameters for two cameras.
    
    This function is responsible for calculating rectification parameters for 
    set of 2 cameras in stereo setup. To properly calculate this transforms,
    user has to provide camera matrices, distortion coefficients and transform
    between cameras in cartesian space (translation and rotation).
    
    Args:
        left_cam_mat: camera matrix with intrinsic for left camera (fx, fy, cx, fy in 3x3 matrix),
        left_dist_coeffs: distortion coefficients for left camera (k1, k2, p1, p2, k3), 
        right_cam_mat: camera matrix with intrinsic for right camera (fx, fy, cx, fy in 3x3 matrix),
        right_dist_coeffs: distortion coefficients for right camera (k1, k2, p1, p2, k3),
        rot_matrix: rotation part of transform between cameras (3x3 matrix),
        trans_vec: translation part of transform between cameras (3x1 matrix),
        imsize: resolution of images used for calibration.
    
    Returns:
        Tuple with 5 elements:
            - 3x3 rectification transform (rotation matrix) for the left camera.
            - 3x3 rectification transform (rotation matrix) for the right camera.
            - 3x4 projection matrix in the new (rectified) coordinate systems for the left camera,
            - 3x4 projection matrix in the new (rectified) coordinate systems for the right camera.
            - 4x4 disparity-to-depth mapping matrix
    """
    R_left, R_right, P_left, P_right, Q, left_roi, right_roi = cv2.stereoRectify(left_cam_mat,
                                                                                 left_dist_coeffs,
                                                                                 right_cam_mat,
                                                                                 right_dist_coeffs,
                                                                                 imsize,
                                                                                 rot_matrix,
                                                                                 trans_vec,
                                                                                 1,
                                                                                 (0, 0),
                                                                                 flags=0,
                                                                                 )
    return R_left, R_right, P_left, P_right, Q

# Draw CHARUCO board model
plt.figure(figsize=(15, 10))
plt.title('Charuco board')
plt.imshow(board.draw((1280, 720)), cmap='gray')
plt.axis('off')
plt.show()

# Load images paths
left_mono_imgs_paths = sorted(glob.glob(os.path.join(path_to_folder, '*_left_mono.png')))
right_mono_imgs_paths = sorted(glob.glob(os.path.join(path_to_folder, '*_right_mono.png')))
color_imgs_paths = sorted(glob.glob(os.path.join(path_to_folder, '*_color.png')))

# Detect Charuco markers for each camera
print('Detecting Charuco corners on left mono images')
left_mono_corners, left_mono_corners_ids, imsize =                 analyze_charuco(left_mono_imgs_paths, aruco_dict, board)
print('Detecting Charuco corners on right mono images')
right_mono_corners, right_mono_corners_ids, imsize =                 analyze_charuco(right_mono_imgs_paths, aruco_dict, board)
print('Detecting Charuco corners on color images')
color_corners, color_corners_ids, imsize =                 analyze_charuco(color_imgs_paths, aruco_dict, board)

# Sample image with corners drawn for each camera
img = cv2.imread(left_mono_imgs_paths[0])
img_corners = cv2.aruco.drawDetectedCornersCharuco(img, left_mono_corners[0], 
                                                   left_mono_corners_ids[0])
plt.figure(figsize=(10, 15))
plt.title('Sample image with drawn detected markers')
plt.imshow(img_corners)
plt.axis('off')
plt.show()

# Calibrate camera using detected corners for each camera
print('Calibrate left mono camera')
left_mono_repr_err, left_mono_cam_mat, left_mono_dist_coeff =             calibrate_camera(left_mono_corners, left_mono_corners_ids, imsize)
print('Calibrate right mono camera')
right_mono_repr_err, right_mono_cam_mat, right_mono_dist_coeff =             calibrate_camera(right_mono_corners, right_mono_corners_ids, imsize)
print('Calibrate color camera')
color_repr_err, color_cam_mat, color_dist_coeff =             calibrate_camera(color_corners, color_corners_ids, imsize)

print('\nLeft mono reprojection error:', left_mono_repr_err)
print('\nRight mono reprojection error:', right_mono_repr_err)
print('\nColor reprojection error:', color_repr_err)
print('\nLeft mono distortion coefficients:', left_mono_dist_coeff, sep='\n')
print('\nRight mono distortion coefficients:', right_mono_dist_coeff, sep='\n')
print('\nColor distortion coefficients:', color_dist_coeff, sep='\n')

# Calibrate stereo between left and right mono camera
print('Calibrate left mono to right mono cameras')
repr_erro_l_r, rot_l_r, trans_l_r =             calibrate_stereo_camera(left_mono_corners, left_mono_corners_ids, 
                                    right_mono_corners, right_mono_corners_ids,
                                    imsize, left_mono_cam_mat, left_mono_dist_coeff, 
                                    right_mono_cam_mat, right_mono_dist_coeff,
                                    board, acceptance_threshold=acceptance_threshold)

# Calibrate stereo between left and color
print('Calibrate left mono to color cameras')
repr_erro_l_color, rot_l_color, trans_l_color =             calibrate_stereo_camera(left_mono_corners, left_mono_corners_ids, 
                                    color_corners, color_corners_ids,
                                    imsize, left_mono_cam_mat, left_mono_dist_coeff, 
                                    color_cam_mat, color_dist_coeff,
                                    board, acceptance_threshold=acceptance_threshold)

with np.printoptions(suppress=True):
    print('\nLeft mono camera matrix:', left_mono_cam_mat, sep='\n')
    print('\nRight mono camera matrix:', right_mono_cam_mat, sep='\n')
    print('\nColor camera matrix:', color_cam_mat, sep='\n')

print('Calculate rectification map for left mono to right mono')
R_left_l_r, R_right_l_r, P_left_l_r, P_right_l_r, Q_l_r = calc_rectification_maps(left_mono_cam_mat, left_mono_dist_coeff, 
                                                                                  right_mono_cam_mat, right_mono_dist_coeff,
                                                                                  rot_l_r, trans_l_r, imsize)

print('Calculate rectification map for left mono to color')
R_left_l_color, R_right_l_color, P_left_l_color, P_right_l_color, Q_l_color = calc_rectification_maps(left_mono_cam_mat, left_mono_dist_coeff, 
                                                                                                      color_cam_mat, color_dist_coeff,
                                                                                                      rot_l_color, trans_l_color, imsize)
        
# Save calibration results to file
left_mono_intrinsic = {"fx": left_mono_cam_mat[0][0],
                       "fy": left_mono_cam_mat[1][1],
                       "cx": left_mono_cam_mat[0][2],
                       "cy": left_mono_cam_mat[1][2],
                       "k1": left_mono_dist_coeff[0][0],
                       "k2": left_mono_dist_coeff[1][0],
                       "p1": left_mono_dist_coeff[2][0],
                       "p2": left_mono_dist_coeff[3][0],
                       "k3": left_mono_dist_coeff[4][0]}

right_mono_intrinsic = {"fx": right_mono_cam_mat[0][0],
                        "fy": right_mono_cam_mat[1][1],
                        "cx": right_mono_cam_mat[0][2],
                        "cy": right_mono_cam_mat[1][2],
                        "k1": right_mono_dist_coeff[0][0],
                        "k2": right_mono_dist_coeff[1][0],
                        "p1": right_mono_dist_coeff[2][0],
                        "p2": right_mono_dist_coeff[3][0],
                        "k3": right_mono_dist_coeff[4][0]}

color_intrinsic = {"fx": color_cam_mat[0][0],
                   "fy": color_cam_mat[1][1],
                   "cx": color_cam_mat[0][2],
                   "cy": color_cam_mat[1][2],
                   "k1": color_dist_coeff[0][0],
                   "k2": color_dist_coeff[1][0],
                   "p1": color_dist_coeff[2][0],
                   "p2": color_dist_coeff[3][0],
                   "k3": color_dist_coeff[4][0]}

# Left mono to right mono
rotation = R.from_matrix(rot_l_r)
quat1 = rotation.as_quat()
left_mono_to_right_mono_extrinsic = {"T": {"x": trans_l_r[0][0],
                                           "y": trans_l_r[1][0],
                                           "z": trans_l_r[2][0]},
                                     "R": {"x": quat1[0],
                                           "y": quat1[1],
                                           "z": quat1[2],
                                           "w": quat1[3]},
                                     "parent": "left_mono",
                                     "child": "right_mono"}
print('\nLeft mono to right mono transform:', json.dumps(left_mono_to_right_mono_extrinsic, indent=4), sep='\n')

# Left mono to color
rotation = R.from_matrix(rot_l_color)
quat1 = rotation.as_quat()
left_mono_to_color_extrinsic = {"T": {"x": trans_l_color[0][0],
                                      "y": trans_l_color[1][0],
                                      "z": trans_l_color[2][0]},
                                "R": {"x": quat1[0],
                                      "y": quat1[1],
                                      "z": quat1[2],
                                      "w": quat1[3]},
                                "parent": "left_mono",
                                "child": "color"}
print('\nLeft mono to color transform:', json.dumps(left_mono_to_color_extrinsic, indent=4), sep='\n')

# Rectification left mono to right mono
rect_left_mono_to_right_mono = {'R_left': R_left_l_r.tolist(),
                                'R_right': R_right_l_r.tolist(),
                                'P_left': P_left_l_r.tolist(),
                                'P_right': P_right_l_r.tolist(),
                                'Q': Q_l_r.tolist(),
                                "parent": "left_mono",
                                "child": "right_mono"}

# Rectification left mono to color
rect_left_mono_to_color = {'R_left': R_left_l_color.tolist(),
                           'R_right': R_right_l_color.tolist(),
                           'P_left': P_left_l_color.tolist(),
                           'P_right': P_right_l_color.tolist(),
                           'Q': Q_l_color.tolist(),
                           'parent': 'left_mono',
                           'child': 'color'}


output_json = {
    "intrinsic": 
    {
        "left_mono": left_mono_intrinsic, 
        "right_mono": right_mono_intrinsic,
        "color": color_intrinsic,
    },
    "extrinsic": 
    [
        left_mono_to_right_mono_extrinsic, 
        left_mono_to_color_extrinsic,
    ],
    'rect':
    [
        rect_left_mono_to_right_mono,
        rect_left_mono_to_color
    ],
}

with open(output_file, 'w') as f:
    json.dump(output_json, f, indent=4)
print(f'Calibration parameters save to: "{output_file}"')
