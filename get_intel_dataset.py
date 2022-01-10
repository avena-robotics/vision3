import os
import sys
import cv2
import time
import numpy as np
import argparse 
sys.path.append(os.path.join(os.path.expanduser('~'), 'vision3'))
from intelcam.open_intel_camera import open_intel_camera
from intelcam.close_intel_camera import close_intel_camera
from intelcam.get_intel_all_images import get_intel_all_images



if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('cam0', help='id of the first camera')
        parser.add_argument('cam1', help='id of the second camera')
        parser.add_argument('base_dir', help='directory where images will be saved to')
        args = parser.parse_args()
        base_dir = args.base_dir

        cam0_handle, cam0_info = open_intel_camera(f'/home/avena/vision3/intelcam/config/cam{args.cam0}_config.json')
        cam1_handle, cam1_info = open_intel_camera(f'/home/avena/vision3/intelcam/config/cam{args.cam1}_config.json')

        os.makedirs(base_dir, exist_ok=True)

        key = input('Press ENTER to continue or "q" and then press ENTER to exit: ')
        while key.lower() != 'q':
            color0, _ = get_intel_all_images(cam0_handle)
            color1, _ = get_intel_all_images(cam1_handle)
            
            ts = time.time_ns()
            cv2.imwrite(f'{base_dir}/{ts}_cam{args.cam0}.png', cv2.cvtColor(color0, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f'{base_dir}/{ts}_cam{args.cam1}.png', cv2.cvtColor(color1, cv2.COLOR_RGB2BGR))

            key = input('Press ENTER to continue or "q" and then press ENTER to exit: ')
    except Exception as e:
        print(f'[ERROR]: {e}')

    close_intel_camera(cam0_handle)
    close_intel_camera(cam1_handle)
