from basler_client_functions.get_basler_all_images import get_basler_all_images

left_mono, color, right_mono = get_basler_all_images()
if left_mono.size != 0 and color.size != 0 and right_mono.size != 0:
    print('Images read successfully')
