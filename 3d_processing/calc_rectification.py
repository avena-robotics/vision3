import cv2
import json
import numpy as np
import time
import matplotlib.pyplot as plt


def calc_rectification(image: np.ndarray, stereo_map_x, stereo_map_y) -> np.ndarray:
    """Rectifies image using provided mappings.

    This function is responsible for rectifying input image using calculated
    mappings in X and Y directions.

    Args:
        image: 1 or 3 channel unrectified image (np.ndarray),
        stereo_map_x: mapping in X direction to undistort and rectify image, mapping is provided for each pixel (np.ndarray),
        stereo_map_y: mapping in Y direction to undistort and rectify image, mapping is provided for each pixel (np.ndarray),

    Returns:
        rectified image with the same shape and type as input image (np.ndarray) 
    """
    img_rect = cv2.remap(image, stereo_map_x, stereo_map_y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
    return img_rect


def init_rect_maps(cam_matrix, dist_coeffs, R_mat, P_mat, imsize):
    """Calculate undistortion and rectification maps.
    
    This function is responsible for calculating maps used to rectify image.
    To calculate those mappings intrinsic parameters, distortion coefficients
    and rotation and projection matrices are used.

    Args:
        cam_matrix: intrinsic parameters (fx, fy, cx, cy) arranged in 3x3 matrix like:
            [[fx,  0, cx], 
             [ 0, fy, cy], 
             [ 0,  0,  1]], type: np.ndarray
        dist_coeffs: distortion coefficients (k1, k2, p1, p2, k3) arrange in 1D array (np.ndarray),
        R_mat: 3x3 rectification transform (rotation matrix) for camera (np.ndarray),
        P_mat: 3x4 projection matrix in the new coordinate system (np.ndarray),
        imsize: resolution of image for which the mapping are calulated, it can be 2 or 3 size tuple (height, width)

    Returns:
        Tuple with 2 elements:
            - stereo_map_x: mapping in X direction to undistort and rectify image, mapping is provided for each pixel (np.ndarray),
            - stereo_map_y: mapping in Y direction to undistort and rectify image, mapping is provided for each pixel (np.ndarray), 
    """
    # OpenCV assumes that image is grayscale and order of shape 
    # is [width, height], so first take only height and width
    # and then reverse order
    image_size = imsize[:2][::-1]
    stereo_map_x, stereo_map_y = cv2.initUndistortRectifyMap(cam_matrix, dist_coeffs, 
                                                             R_mat, P_mat, image_size, cv2.CV_16SC2)
    return stereo_map_x, stereo_map_y


def get_rectification_params(path: str, first_cam: str, second_cam: str):
    with open(path, 'r') as f:
        input_dict = json.load(f)

    temp = input_dict["intrinsic"][first_cam]
    left_cam_mat = np.eye(3)
    left_cam_mat[0][0] = temp["fx"]
    left_cam_mat[1][1] = temp["fy"]
    left_cam_mat[0][2] = temp["cx"]
    left_cam_mat[1][2] = temp["cy"]
    left_dist_coeffs = np.array([temp["k1"], temp["k2"], temp["p1"], temp["p2"], temp["k3"]])
    left_imsize = temp["height"], temp["width"] 

    temp = input_dict["intrinsic"][second_cam]
    right_cam_mat = np.eye(3)
    right_cam_mat[0][0] = temp["fx"]
    right_cam_mat[1][1] = temp["fy"]
    right_cam_mat[0][2] = temp["cx"]
    right_cam_mat[1][2] = temp["cy"]
    right_dist_coeffs = np.array([temp["k1"], temp["k2"], temp["p1"], temp["p2"], temp["k3"]])
    right_imsize = temp["height"], temp["width"] 

    temp = input_dict["rect"]
    for ext in temp:
        if ext["first_cam"] == first_cam and ext["second_cam"] == second_cam:
            P_left = np.array(ext['P_left'])
            P_right = np.array(ext['P_right'])
            R_left = np.array(ext['R_left'])
            R_right = np.array(ext['R_right'])
            break
    else:
        raise RuntimeError(f'There are no rectification parameters between "{first_cam}" and "{second_cam}"')
    return left_cam_mat, left_dist_coeffs, right_cam_mat, right_dist_coeffs, P_left, P_right, R_left, R_right, left_imsize, right_imsize


if __name__ == '__main__':
    #######################################################
    # Configuration
    first_camera_name = 'left_mono'
    second_camera_name = 'right_mono'
    calib_config_path = '/home/avena/vision3/calib/basler_calibration.json'
    left_image_path = f'/home/avena/left_mono.png'
    right_image_path = f'/home/avena/right_mono.png'

    left_image_rect_out_path = '/home/avena/vision3/3d_processing/first_image_rect.png'
    right_image_rect_out_path = '/home/avena/vision3/3d_processing/second_image_rect.png'
    #######################################################

    # Load calibration parameters
    left_cam_mat, left_dist_coeffs, right_cam_mat, right_dist_coeffs, left_P, right_P, left_R, right_R, left_imsize, right_imsize = get_rectification_params(calib_config_path, first_camera_name, second_camera_name)

    # Initialize remappings
    left_stereo_map_x, left_stereo_map_y = init_rect_maps(left_cam_mat, left_dist_coeffs, 
                                                          left_R, left_P, left_imsize)
    right_stereo_map_x, right_stereo_map_y = init_rect_maps(right_cam_mat, right_dist_coeffs, 
                                                            right_R, right_P, right_imsize)

    # Load unrectified images
    left_image = cv2.imread(left_image_path)
    right_image = cv2.imread(right_image_path)

    # Execute rectification function
    start = time.perf_counter()
    left_image_rect = calc_rectification(left_image, left_stereo_map_x, left_stereo_map_y)
    print(f'Rectification time (one image): {(time.perf_counter() - start) * 1000} [ms]')
    right_image_rect = calc_rectification(right_image, right_stereo_map_x, right_stereo_map_y)

    # Save rectified images to PNG
    cv2.imwrite(left_image_rect_out_path, left_image_rect)
    cv2.imwrite(right_image_rect_out_path, right_image_rect)
    print('Rectified left and right images are saved')
