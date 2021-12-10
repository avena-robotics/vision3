import cv2
from basler_client_functions.get_basler_all_images import get_basler_all_images

# First, make sure that cameras are opened

# Getting images from controller
left_mono, color, right_mono = get_basler_all_images()
if left_mono.size != 0 and color.size != 0 and right_mono.size != 0:
    print('Images read successfully')
else:
    print('[ERROR]: Cannot read images from camera controller')
    exit(1)

# Saving images to PNGs
cv2.imwrite('left_mono.png', left_mono)
cv2.imwrite('right_mono.png', right_mono)
cv2.imwrite('color.png', color)
