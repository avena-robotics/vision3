from basler_client_functions.open_basler_cameras import open_basler_cameras
from basler_client_functions.close_basler_cameras import close_basler_cameras
from basler_client_functions.get_basler_all_images import get_basler_all_images
from basler_client_functions.get_basler_color_image import get_basler_color_image
from basler_client_functions.get_basler_mono_images import get_basler_mono_images

import cv2
import time
import os

#########################################################
# User modifies path for dataset
BASE_DIR = '/home/avena/vision3/calib/dataset001'
#########################################################


os.makedirs(BASE_DIR, exist_ok=True)

opened_cameras = open_basler_cameras()
if not opened_cameras:
    print('Cannot open Basler cameras. Exiting...')
    exit(1)

print('Opened cameras successfully')
pressed_key = input('Pressed ENTER to make a photo (or "q" and then ENTER to exit): ')
cnt = 1
while pressed_key != 'q':
    left_mono, color, right_mono = get_basler_all_images()    
    if left_mono.size == 0 or color.size == 0 or right_mono.size == 0:
        print('Failed to get images. Try again...')
        continue
    ts_now = time.time_ns()
    cv2.imwrite(os.path.join(BASE_DIR, f'{ts_now}_left_mono.png'), left_mono)
    cv2.imwrite(os.path.join(BASE_DIR, f'{ts_now}_right_mono.png'), right_mono)
    cv2.imwrite(os.path.join(BASE_DIR, f'{ts_now}_color.png'), color)
    print(f'Save images called {cnt} times')
    cnt += 1
    pressed_key = input('Pressed ENTER to make a photo (or "q" and then ENTER to exit): ')

print('Closing cameras. Success:', close_basler_cameras())