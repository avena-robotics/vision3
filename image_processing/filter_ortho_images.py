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


if __name__ == '__main__':
    ##################################################
    # User configuration
    image_path = '/home/avena/Dropbox/captures/003-intel-measuremement/1640173716969686089_color.png'
    ksize = 5 # filter size
    out_image_path = 'filtered.png'
    ##################################################

    img = cv2.imread(image_path)
    holes_filled = filter_black_holes(img, ksize)
    cv2.imwrite(out_image_path, holes_filled)
