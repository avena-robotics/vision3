import cv2
from basler_client_functions.get_basler_color_image import get_basler_color_image

# First, make sure that cameras are opened

# Get single color image
color = get_basler_color_image()
if color.size != 0:
  	print('Color image read successfully')
else:
    print('[ERROR]: Cannot read color image from camera controller')
    exit(1)

# Saving images to PNGs
cv2.imwrite('color.png', color)
