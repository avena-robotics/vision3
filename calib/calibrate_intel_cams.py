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




if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('cam0', help='id of the target camera', type=int)
        parser.add_argument('cam1', help='id of the source camera', type=int)
        parser.add_argument('base_dir', help='directory where images will be read from', type=str)
        parser.add_argument('-t', '--threshold', help='minimal amount of common samples between cameras', default=60, type=int)
        args = parser.parse_args()

        cam0_handle, cam0_info = open_intel_camera(f'/home/avena/vision3/intelcam/config/cam{args.cam0}_config.json')
        cam1_handle, cam1_info = open_intel_camera(f'/home/avena/vision3/intelcam/config/cam{args.cam1}_config.json')
        close_intel_camera(cam0_handle)
        close_intel_camera(cam1_handle)

        # Load images paths
        cam0_imgs_paths = sorted(glob.glob(os.path.join(f'{args.base_dir}', f'*_cam{args.cam0}.png')))
        cam1_imgs_paths = sorted(glob.glob(os.path.join(f'{args.base_dir}', f'*_cam{args.cam1}.png')))

        acceptance_threshold = args.threshold  # minimal number of detected markers 
                                            # on each image when calibrating in stereo

        # Aruco dictionary definition 
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)

        # Charuco board definition
        board = cv2.aruco.CharucoBoard_create(squaresX=12, 
                                            squaresY=9, 
                                            squareLength=0.06,  # meters 
                                            markerLength=0.047, # meters
                                            dictionary=aruco_dict)

        # Detect Charuco markers for each camera
        print(f'Detecting Charuco corners on cam {args.cam0} images')
        cam0_corners, cam0_corners_ids, imsize = analyze_charuco(cam0_imgs_paths, aruco_dict, board)
        print(f'Detecting Charuco corners on cam {args.cam1} images')
        cam1_corners, cam1_corners_ids, imsize = analyze_charuco(cam1_imgs_paths, aruco_dict, board)

        # Calibrate stereo between left and right mono camera
        print('Calibrate cameras')
        cam1_cam_mat = np.array(
            [[cam1_info['fx'], 0, cam1_info['cx']],
            [0, cam1_info['fy'], cam1_info['cy']],
            [0, 0, 1]]
        )
        
        cam0_cam_mat = np.array(
            [[cam0_info['fx'], 0, cam0_info['cx']],
            [0, cam0_info['fy'], cam0_info['cy']],
            [0, 0, 1]]
        )

        print(f'Calibrate cam {args.cam1} to cam {args.cam0}')
        repr_erro_cam1_cam0, rot_cam1_cam0, trans_cam1_cam0 = calibrate_stereo_camera(cam1_corners, cam1_corners_ids, 
                                            cam0_corners, cam0_corners_ids,
                                            imsize, cam1_cam_mat, np.zeros((5, 1)), 
                                            cam0_cam_mat, np.zeros((5, 1)),
                                            board, acceptance_threshold=acceptance_threshold)


        print(f'Calibration data for cam {args.cam1} to cam {args.cam0}')
        transform0 = np.zeros((4, 4))
        transform0[:3, :3] = rot_cam1_cam0
        transform0[:, 3] = [trans_cam1_cam0[0][0], trans_cam1_cam0[1][0], trans_cam1_cam0[2][0], 1]
        print(repr(transform0))

        calib_file_path = f'charuco_calibration_cam{args.cam1}_to_cam{args.cam0}.json'
        print(f'Saving calibration to "{calib_file_path}"')
        with open(calib_file_path, 'w') as f:
            json.dump(transform0.tolist(), f, indent=3)
    except Exception as e:
        print(f'[ERROR]:', e)
