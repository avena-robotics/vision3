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
        parser.add_argument('cam', help='id of the target camera', type=int)
        args = parser.parse_args()

        cam_id = args.cam
        cam0_handle, cam0_info = open_intel_camera(f'/home/avena/vision3/intelcam/config/cam{cam_id}_config.json')

        # Aruco dictionary definition 
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)

        # Charuco board definition
        board = cv2.aruco.CharucoBoard_create(squaresX=12, 
                                            squaresY=9, 
                                            squareLength=0.06,  # meters 
                                            markerLength=0.047, # meters
                                            dictionary=aruco_dict)

        color, _ = get_intel_all_images(cam0_handle)
        cv2.imwrite('camera_to_plane.png', color)

        # Detect Charuco markers for each camera
        print(f'Detecting Charuco corners on cam {cam_id} images')
        cam0_corners, cam0_corners_ids, imsize = analyze_charuco(['camera_to_plane.png'], aruco_dict, board)
        os.remove('camera_to_plane.png')

        # Calibrate stereo between left and right mono camera
        print('Calibrate cameras')
        cam0_cam_mat = np.array(
            [[cam0_info['fx'], 0, cam0_info['cx']],
            [0, cam0_info['fy'], cam0_info['cy']],
            [0, 0, 1]]
        )

        ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(cam0_corners[0], cam0_corners_ids[0], board, cam0_cam_mat, np.zeros((5, 1)), None, None)

        rot_matrix, jac = cv2.Rodrigues(rvec)
        rot_matrix = rot_matrix @ np.array([[1.0000000,  0.0000000,  0.0000000],
                                            [0.0000000,  -1.0000000,  0.0000000],
                                            [0.0000000,  0.0000000,  -1.0000000]])
        transform = np.identity(4)
        transform[:3, :3] = rot_matrix
        print('Rotation:', repr(rot_matrix), sep='\n')
        print('Transform:', repr(transform), sep='\n')

        # if ret:
        #     image = cv2.aruco.drawDetectedCornersCharuco(color, cam0_corners[0], cam0_corners_ids[0], (255, 0, 0))
        #     image = cv2.aruco.drawAxis(image, cam0_cam_mat, np.zeros((5, 1)), rvec, tvec, 0.1)
        #     cv2.imshow('kek', image)
        #     cv2.waitKey()

        calib_file_path = f'calibration_cam{cam_id}_to_table.json'
        print(f'Saving calibration to table: "{calib_file_path}"')
        with open(calib_file_path, 'w') as f:
            json.dump(transform.tolist(), f)

    except Exception as e:
        print('[ERROR]:', e)

    close_intel_camera(cam0_handle)
