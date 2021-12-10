import cv2
import json
import numpy as np


def get_perspective_transform(path: str, first_cam: str, second_cam: str) -> np.ndarray:
    with open(path, 'r') as f:
        input_dict = json.load(f)
    temp = input_dict["rect"]
    for ext in temp:
        if ext["first_cam"] == first_cam and ext["second_cam"] == second_cam:
            Q = np.array(ext['Q'])
            break
    else:
        raise RuntimeError(f'There are no rectification parameters between "{first_cam}" and "{second_cam}"')
    return Q


def calc_depth(disparity: np.ndarray, q: np.ndarray) -> np.ndarray:
    q[3, 2] *= -1
    image_3d = cv2.reprojectImageTo3D(disparity, q, handleMissingValues=True)


    # #############################################
    # # TODO: Testing
    # print(image_3d)
    # print(image_3d.shape)
    # image_3d_cp = image_3d.copy()
    # valid_mask = image_3d_cp[:, :, 2] != image_3d_cp[:, :, 2].max()
    # image_3d_cp = image_3d_cp[valid_mask]
    # save_point_cloud_to_obj('cloud.obj', image_3d_cp)
    # #############################################

    image_z = image_3d[:, :, 2]
    invalid_mask = image_z == image_z.max()
    image_z[invalid_mask] = 0
    image_z *= -1

    return image_z


def save_point_cloud_to_obj(path:str, points: np.ndarray):
    """Saves point cloud to OBJ file
    
    Args:
        path: system path where the point cloud will be saved,
        points: NR_POINTSx3 array with 3D coordinates of point cloud        
    """
    points_cp = points.copy().reshape(-1, 3)
    with open(path, 'w') as f:
        for p in points_cp:
            f.write(f'v {p[0]} {p[1]} {p[2]}\n')


if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    #######################################################
    # Configuration
    disparity_path = '/home/avena/vision3/3d_processing/calc_disparity/disparity.png'
    calib_config_path = '/home/avena/vision3/calib/basler_calibration.json'

    depth_out_path = '/home/avena/vision3/3d_processing/calc_depth/depth.png'
    #######################################################

    # Loading disparity image from file
    disparity = cv2.imread(disparity_path, cv2.IMREAD_UNCHANGED)

    # Reading Q matrix (perspective transformation matrix from calibration results)
    q = get_perspective_transform(calib_config_path, 'left_mono', 'right_mono')

    # Calculate depth using disparity and Q matrix
    depth, image_3d = calc_depth(disparity, q)
    # print(f'Calculated depth save to "{depth_out_path}"')
