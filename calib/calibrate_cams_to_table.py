import os
import sys
import cv2
import glob
import json
from scipy.spatial.transform import Rotation as R
import numpy as np
sys.path.append(os.path.join(os.path.expanduser('~'), 'vision3'))
from intelcam.open_intel_camera import open_intel_camera
from intelcam.close_intel_camera import close_intel_camera
from intelcam.get_intel_all_images import get_intel_all_images
from calib.calibrate import analyze_charuco, calibrate_stereo_camera
import argparse
np.set_printoptions(suppress=True)


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # parser.add_argument('cam', help='id of the target camera', type=int)
        # args = parser.parse_args()

        cam1_handle, cam1_info = open_intel_camera(f'/home/avena/vision3/intelcam/config/cam1_config.json')
        cam4_handle, cam4_info = open_intel_camera(f'/home/avena/vision3/intelcam/config/cam4_config.json')
        cam5_handle, cam5_info = open_intel_camera(f'/home/avena/vision3/intelcam/config/cam5_config.json')

        # Aruco dictionary definition 
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)

        # Charuco board definition
        board = cv2.aruco.CharucoBoard_create(squaresX=12, 
                                            squaresY=9, 
                                            squareLength=0.06,  # meters 
                                            markerLength=0.047, # meters
                                            dictionary=aruco_dict)

        rvec1_sum = np.zeros(shape=(3, 1))
        tvec1_sum = np.zeros(shape=(3, 1))
        rvec4_sum = np.zeros(shape=(3, 1))
        tvec4_sum = np.zeros(shape=(3, 1))
        rvec5_sum = np.zeros(shape=(3, 1))
        tvec5_sum = np.zeros(shape=(3, 1))
        
        NUMBER_OF_ITERATIONS = 10
        for i in range(NUMBER_OF_ITERATIONS):
                
            color1, _ = get_intel_all_images(cam1_handle)
            color4, _ = get_intel_all_images(cam4_handle)
            color5, _ = get_intel_all_images(cam5_handle)
            cv2.imwrite('cam1_to_plane.png', color1)
            cv2.imwrite('cam4_to_plane.png', color4)
            cv2.imwrite('cam5_to_plane.png', color5)

            # Detect Charuco markers for each camera
            print(f'Detecting Charuco corners on cam images')
            cam1_corners, cam1_corners_ids, imsize = analyze_charuco(['cam1_to_plane.png'], aruco_dict, board)
            cam4_corners, cam4_corners_ids, imsize = analyze_charuco(['cam4_to_plane.png'], aruco_dict, board)
            cam5_corners, cam5_corners_ids, imsize = analyze_charuco(['cam5_to_plane.png'], aruco_dict, board)
            os.remove('cam1_to_plane.png')
            os.remove('cam4_to_plane.png')
            os.remove('cam5_to_plane.png')

            # Calibrate stereo between left and right mono camera
            print('Calibrate cameras')
            cam1_cam_mat = np.array(
                [[cam1_info['fx'], 0, cam1_info['cx']],
                [0, cam1_info['fy'], cam1_info['cy']],
                [0, 0, 1]]
            )
            cam4_cam_mat = np.array(
                [[cam4_info['fx'], 0, cam4_info['cx']],
                [0, cam4_info['fy'], cam4_info['cy']],
                [0, 0, 1]]
            )
            cam5_cam_mat = np.array(
                [[cam5_info['fx'], 0, cam5_info['cx']],
                [0, cam5_info['fy'], cam5_info['cy']],
                [0, 0, 1]]
            )

            ret1, rvec1, tvec1 = cv2.aruco.estimatePoseCharucoBoard(cam1_corners[0], cam1_corners_ids[0], board, cam1_cam_mat, np.zeros((5, 1)), None, None)
            ret4, rvec4, tvec4 = cv2.aruco.estimatePoseCharucoBoard(cam4_corners[0], cam4_corners_ids[0], board, cam4_cam_mat, np.zeros((5, 1)), None, None)
            ret5, rvec5, tvec5 = cv2.aruco.estimatePoseCharucoBoard(cam5_corners[0], cam5_corners_ids[0], board, cam5_cam_mat, np.zeros((5, 1)), None, None)
            rvec1_sum += rvec1
            tvec1_sum += tvec1
            rvec4_sum += rvec4
            tvec4_sum += tvec4
            rvec5_sum += rvec5
            tvec5_sum += tvec5
        
        rvec1 = rvec1_sum / NUMBER_OF_ITERATIONS
        tvec1 = tvec1_sum / NUMBER_OF_ITERATIONS
        rvec4 = rvec4_sum / NUMBER_OF_ITERATIONS
        tvec4 = tvec4_sum / NUMBER_OF_ITERATIONS
        rvec5 = rvec5_sum / NUMBER_OF_ITERATIONS
        tvec5 = tvec5_sum / NUMBER_OF_ITERATIONS


        rot_matrix1, _ = cv2.Rodrigues(rvec1)
        transform1 = np.identity(4)
        transform1[:3, :3] = rot_matrix1
        transform1[:3, 3] = tvec1.flatten()
        print('Transform:', repr(transform1), sep='\n')
        calib_file_path = f'calibration_cam1_to_table.json'
        print(f'Saving calibration to table: "{calib_file_path}"')
        with open(calib_file_path, 'w') as f:
            json.dump(transform1.tolist(), f)

        rot_matrix4, _ = cv2.Rodrigues(rvec4)
        transform4 = np.identity(4)
        transform4[:3, :3] = rot_matrix4
        transform4[:3, 3] = tvec4.flatten()
        print('Transform:', repr(transform4), sep='\n')
        calib_file_path = f'calibration_cam4_to_table.json'
        print(f'Saving calibration to table: "{calib_file_path}"')
        with open(calib_file_path, 'w') as f:
            json.dump(transform4.tolist(), f)

        rot_matrix5, _ = cv2.Rodrigues(rvec5)
        transform5 = np.identity(4)
        transform5[:3, :3] = rot_matrix5
        transform5[:3, 3] = tvec5.flatten()
        print('Transform:', repr(transform5), sep='\n')
        calib_file_path = f'calibration_cam5_to_table.json'
        print(f'Saving calibration to table: "{calib_file_path}"')
        with open(calib_file_path, 'w') as f:
            json.dump(transform5.tolist(), f)
        
        if ret1 and ret4 and ret5:
            image1 = cv2.aruco.drawDetectedCornersCharuco(color1, cam1_corners[0], cam1_corners_ids[0], (255, 0, 0))
            image1 = cv2.aruco.drawAxis(image1, cam1_cam_mat, np.zeros((5, 1)), rvec1, tvec1, 0.1)

            image4 = cv2.aruco.drawDetectedCornersCharuco(color4, cam4_corners[0], cam4_corners_ids[0], (255, 0, 0))
            image4 = cv2.aruco.drawAxis(image4, cam4_cam_mat, np.zeros((5, 1)), rvec4, tvec4, 0.1)

            image5 = cv2.aruco.drawDetectedCornersCharuco(color5, cam5_corners[0], cam5_corners_ids[0], (255, 0, 0))
            image5 = cv2.aruco.drawAxis(image5, cam5_cam_mat, np.zeros((5, 1)), rvec5, tvec5, 0.1)
            
            cv2.imshow('cam1', image1)
            cv2.imshow('cam4', image4)
            cv2.imshow('cam5', image5)
            cv2.waitKey()

    except Exception as e:
        print('[ERROR]:', e)

    close_intel_camera(cam1_handle)
    close_intel_camera(cam4_handle)
    close_intel_camera(cam5_handle)
