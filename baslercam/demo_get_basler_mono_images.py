import cv2
from basler_client_functions.get_basler_mono_images import get_basler_mono_images

# First, make sure that cameras are opened

# Getting left mono and right mono images from controller
left_mono, right_mono = get_basler_mono_images()
if left_mono.size != 0 and right_mono.size != 0:
    print('Mono images read successfully')
else:
    print('[ERROR]: Cannot read left mono and right mono images from camera controller')
    exit(1)

# Saving images to PNGs
cv2.imwrite('left_mono.png', left_mono)
cv2.imwrite('right_mono.png', right_mono)
