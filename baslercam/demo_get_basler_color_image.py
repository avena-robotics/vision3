from basler_client_functions.get_basler_color_image import get_basler_color_image

color = get_basler_color_image()
if color.size != 0:
  	print('Color image read successfully')
