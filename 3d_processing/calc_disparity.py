import time
import cv2
import numpy as np


def calc_disparity(left_image: np.ndarray, right_image: np.ndarray, win_size=5, min_disparity=0, max_disparity=256) -> np.ndarray:
    """Calculated disparity image.

    This function is responsible for calculating disparity image
    using two rectified images using SGBM (semi global block matching)
    algorithm

    Args:
        left_image: rectified left image (np.ndarray),
        right_image: rectified right image (np.ndarray),
        win_size: windows size which should be used to find matching pixels (default: 5),
        min_disparity: minimum possible disparity value (default: 0),
        max_disparity: maximum allowed disparity (for higher resolutions e.g. 4K
                       this value should be at least 512) (default: 256),
    
    Returns:
        disparity image in range (-1, max_disparity - 1)
    """
    stereo = cv2.StereoSGBM_create(minDisparity=min_disparity,
                                   numDisparities=max_disparity-min_disparity,
                                   blockSize=win_size,
                                   uniquenessRatio=5,
                                   speckleWindowSize=5,
                                   speckleRange=5,
                                   disp12MaxDiff=1,
                                   P1=8*3*win_size**2,#8*3*win_size**2,
                                   P2=32*3*win_size**2) #32*3*win_size**2
    disparity_map = stereo.compute(left_image, right_image).astype(np.float32) / 16  # divide by 16 to remove 4 fractional bits (as of documentation) 2^4 = 16 
    return disparity_map



if __name__ == '__main__':
    #######################################################
    # Configuration
    left_image_path = '/home/avena/vision3/3d_processing/first_image_rect.png'
    right_image_path = '/home/avena/vision3/3d_processing/second_image_rect.png'
    max_disparity = 512

    disparity_out_path = '/home/avena/vision3/3d_processing/disparity.png'
    #######################################################

    # Loading images
    left_image_rect = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_image_rect = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

    # Execute disparity
    start = time.perf_counter()
    disparity = calc_disparity(left_image_rect, right_image_rect, max_disparity=max_disparity)
    print(f'Calculating disparity time: {(time.perf_counter() - start) * 1000} [ms]')
    
    # Save results
    disparity_cp = disparity.copy()
    disparity_cp[disparity_cp < 0] = 0
    cv2.imwrite(disparity_out_path, ((disparity_cp / disparity_cp.max()) * 255).astype(np.uint8))
    print(f'Disparity saved to {disparity_out_path}')
