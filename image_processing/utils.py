import numpy as np
import cv2


def filter_black_holes(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Applies median filter only for black pixels for RGB image
    """
    black_pixel_mask = (image == [0, 0, 0]).astype(np.uint8)
    median_filtered = cv2.medianBlur(image, ksize)
    black_filtered = median_filtered * black_pixel_mask
    final_result = image + black_filtered
    return final_result


def colormap_image(image: np.ndarray) -> np.ndarray:
    """Colorize depth image

    This function is responsible for coloring single channel image using
    JET colormap. Only pixel values which are not zero are colormapped.
    This adaptive colorization enables to see changes even on small range
    of pixel values.  

    Args:
        image: array with pixel values with one channel

    Returns:
        colored image with BGR values
    """

    if len(image.shape) != 2:
        raise ValueError('Input image has to be single channel only')

    invalid_mask = image == 0
    image_f32 = (image / image.max()).astype(np.float32)
    min_non_zero_dist = np.unique(image_f32)[1] 
    image_scaled = image_f32 - min_non_zero_dist
    image_scaled = (image_scaled / image_scaled.max() * 255)
    image_scaled[image_scaled < 0] = 0
    colormap = cv2.applyColorMap(image_scaled.astype(np.uint8), cv2.COLORMAP_JET)
    colormap[invalid_mask] = [0, 0, 0]
    return colormap


# if __name__ == '__main__':
#     ##################################################
#     # User configuration
#     image_path = '/home/avena/Dropbox/captures/003-intel-measuremement/1640173716969686089_color.png'
#     ksize = 5 # filter size
#     out_image_path = 'filtered.png'
#     ##################################################

#     img = cv2.imread(image_path)
#     holes_filled = filter_black_holes(img, ksize)
#     cv2.imwrite(out_image_path, holes_filled)
