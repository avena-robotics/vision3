from basler_client_functions.get_basler_mono_images import get_basler_mono_images

left_mono, right_mono = get_basler_mono_images()
if left_mono.size != 0 and right_mono.size != 0:
    print('Mono images read successfully')
