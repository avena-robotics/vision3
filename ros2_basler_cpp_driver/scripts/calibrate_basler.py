import cv2
from cv2 import aruco
import glob
import json
import numpy as np
import os
from scipy.spatial.transform import Rotation as R


def analyze_charuco(images, aruco_dict, board):

    allCorners = []
    allIds = []
    all_marker_corners = []
    all_marker_ids = []
    all_recovered = []
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.00001)
    for im in images:
        frame = cv2.imread(im)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        marker_corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)
        marker_corners, ids, refusd, recoverd = cv2.aruco.refineDetectedMarkers(gray, board, marker_corners, ids,
                                                                                rejectedCorners=rejectedImgPoints)
        if len(marker_corners) > 0:

            res2 = cv2.aruco.interpolateCornersCharuco(
                marker_corners, ids, gray, board)

            if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3:

                cv2.cornerSubPix(gray, res2[1],
                                 winSize=(5, 5),
                                 zeroZone=(-1, -1),
                                 criteria=criteria)
                allCorners.append(res2[1])
                allIds.append(res2[2])
                all_marker_corners.append(marker_corners)
                all_marker_ids.append(ids)
                all_recovered.append(recoverd)
            else:
                pass
        else:
            pass

    imsize = gray.shape[::-1]
    return allCorners, allIds, all_marker_corners, all_marker_ids, imsize, all_recovered


#######################################################################################################################


def calibrate_camera_charuco(allCorners, allIds, imsize, board):
    """
    Calibrates the camera using the dected corners.
    """
    cameraMatrixInit = np.array([[1920/2, 0.0, 1080/2],
                                 [0.0, 1920/2, 1920/2],
                                 [0.0, 0.0, 1.0]])

    distCoeffsInit = np.zeros((5, 1))
    flags = cv2.CALIB_USE_INTRINSIC_GUESS
    (ret, camera_matrix, distortion_coefficients,
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
        charucoCorners=allCorners,
        charucoIds=allIds,
        board=board,
        imageSize=imsize,
        cameraMatrix=cameraMatrixInit,
        distCoeffs=distCoeffsInit,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

    return ret, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors


#######################################################################################################################


def calibrate_stereo(allCorners_l, allIds_l, allCorners_r, allIds_r, imsize, cameraMatrix_l, distCoeff_l,
                     cameraMatrix_r, distCoeff_r, board, acceptance_threshold=108):
    left_corners_sampled = []
    right_corners_sampled = []
    obj_pts = []
    one_pts = board.chessboardCorners
    for i in range(len(allIds_l)-2):
        left_sub_corners = []
        right_sub_corners = []
        obj_pts_sub = []
        if len(allIds_l[i]) < acceptance_threshold or len(allIds_r[i]) < acceptance_threshold:
            continue
        for j in range(len(allIds_l[i])):
            idx = np.where(allIds_r[i] == allIds_l[i][j])
            if idx[0].size == 0:
                continue
            left_sub_corners.append(allCorners_l[i][j])
            right_sub_corners.append(allCorners_r[i][idx])
            obj_pts_sub.append(one_pts[allIds_l[i][j]])

        obj_pts.append(np.array(obj_pts_sub, dtype=np.float32))
        left_corners_sampled.append(
            np.array(left_sub_corners, dtype=np.float32))
        right_corners_sampled.append(
            np.array(right_sub_corners, dtype=np.float32))

    stereocalib_criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC

    return cv2.stereoCalibrate(
        obj_pts, left_corners_sampled, right_corners_sampled,
        cameraMatrix_l, distCoeff_l, cameraMatrix_r, distCoeff_r, imsize,
        criteria=stereocalib_criteria, flags=flags)


#######################################################################################################################

def calibrate():
    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
    board = aruco.CharucoBoard_create(12, 9, 0.06, 0.045, aruco_dict)
    # Intrinsic for both cameras
    # cam1_ir1 = r"1_new/20211118-*-40093166-mono-image*"
    cam1_ir1 = r"fhd_dataset/l/*.png"
    # cam1_ir2 = r"1_new/20211118-*-40134758-mono-image*"
    cam1_ir2 = r"fhd_dataset/r/*.png"
    cam1_ir1_images = np.array(sorted(glob.glob(cam1_ir1)))
    cam1_ir2_images = np.array(sorted(glob.glob(cam1_ir2)))
    # find charuco patter on each img (INTRINSIC)
    all_corners_cam1_ir1, all_ids_cam1_ir1, all_marker_corners_cam1_ir1, all_marker_ids_cam1_ir1, imsize_cam1_ir1, all_recovered_cam1_ir1 = analyze_charuco(
        cam1_ir1_images, aruco_dict, board)
    all_corners_cam1_ir2, all_ids_cam1_ir2, all_marker_corners_cam1_ir2, all_marker_ids_cam1_ir2, imsize_cam1_ir2, all_recovered_cam1_ir2 = analyze_charuco(
        cam1_ir2_images, aruco_dict, board)
    # find intrinsic parameters
    ret_cam1_ir1, camera_matrix_cam1_ir1, distortion_coefficients_cam1_ir1, rotation_vectors_cam1_ir1, translation_vectors_cam1_ir1 = calibrate_camera_charuco(
        all_corners_cam1_ir1, all_ids_cam1_ir1, imsize_cam1_ir1, board)
    ret_cam1_ir2, camera_matrix_cam1_ir2, distortion_coefficients_cam1_ir2, rotation_vectors_cam1_ir2, translation_vectors_cam1_ir2 = calibrate_camera_charuco(
        all_corners_cam1_ir2, all_ids_cam1_ir2, imsize_cam1_ir2, board)
    # find extrinsic parameters
    ret_cam1_ir1_to_cam1_ir2 = calibrate_stereo(all_corners_cam1_ir1, all_ids_cam1_ir1, all_corners_cam1_ir2,
                                                all_ids_cam1_ir2, imsize_cam1_ir1, camera_matrix_cam1_ir1,
                                                distortion_coefficients_cam1_ir1, camera_matrix_cam1_ir2,
                                                distortion_coefficients_cam1_ir2, board,
                                                acceptance_threshold=50)
    transform_cam1_ir1_to_cam1_ir2 = np.eye(4)
    transform_cam1_ir1_to_cam1_ir2[:3, :3] = ret_cam1_ir1_to_cam1_ir2[5]
    transform_cam1_ir1_to_cam1_ir2[:3, 3] = ret_cam1_ir1_to_cam1_ir2[6].reshape(3, )
    rotation = R.from_matrix(ret_cam1_ir1_to_cam1_ir2[5])
    quat1 = rotation.as_quat()

    # Create intrunsics structures to save
    cam1_ir1_intrinsic = {"fx": camera_matrix_cam1_ir1[0][0],
                          "fy": camera_matrix_cam1_ir1[1][1],
                          "cx": camera_matrix_cam1_ir1[0][2],
                          "cy": camera_matrix_cam1_ir1[1][2],
                          "k1": distortion_coefficients_cam1_ir1[0][0],
                          "k2": distortion_coefficients_cam1_ir1[1][0],
                          "p1": distortion_coefficients_cam1_ir1[2][0],
                          "p2": distortion_coefficients_cam1_ir1[3][0],
                          "k3": distortion_coefficients_cam1_ir1[4][0]}

    cam1_ir2_intrinsic = {"fx": camera_matrix_cam1_ir2[0][0],
                          "fy": camera_matrix_cam1_ir2[1][1],
                          "cx": camera_matrix_cam1_ir2[0][2],
                          "cy": camera_matrix_cam1_ir2[1][2],
                          "k1": distortion_coefficients_cam1_ir2[0][0],
                          "k2": distortion_coefficients_cam1_ir2[1][0],
                          "p1": distortion_coefficients_cam1_ir2[2][0],
                          "p2": distortion_coefficients_cam1_ir2[3][0],
                          "k3": distortion_coefficients_cam1_ir2[4][0]}


    cam1_ir1_to_cam1_ir2_extrinsic = {"T": {"x": ret_cam1_ir1_to_cam1_ir2[6][0][0],
                                            "y": ret_cam1_ir1_to_cam1_ir2[6][1][0],
                                            "z": ret_cam1_ir1_to_cam1_ir2[6][2][0]},
                                      "R": {"x": quat1[0],
                                            "y": quat1[1],
                                            "z": quat1[2],
                                            "w": quat1[3]},
                                      "parent": "cam1_ir1",
                                      "child": "cam1_ir2"}

    output_json = {"intrinsic": {"cam1_ir1": cam1_ir1_intrinsic, "cam1_ir2": cam1_ir2_intrinsic},"extrinsic": [cam1_ir1_to_cam1_ir2_extrinsic]}
    with open(os.path.join('fhd_dataset', 'fhd_basler_calibration.json'), 'w') as f:
        json.dump(output_json, f, indent=4)


####################################################################################################33
def read_custom_json(path: str = "") -> dict:
    with open(path, 'r') as f:
        return json.load(f)

def read_camera_matrix(input_dict: dict, camera_id: str):
    temp = input_dict["intrinsic"][camera_id]
    template = np.eye(3)
    template[0][0] = temp["fx"]
    template[1][1] = temp["fy"]
    template[0][2] = temp["cx"]
    template[1][2] = temp["cy"]
    return template


def read_distortion_vector(input_dict: dict, camera_id: str):
    template = input_dict["intrinsic"][camera_id]
    temp = np.array([template["k1"], template["k2"], template["p1"], template["p2"], template["k3"]])
    return temp


def read_extrinsic(input_dict: dict, parent_id="", child_id=""):
    temp = input_dict["extrinsic"]
    for ext in temp:
      	# Check whether there is requested transformation in input
        if ext["parent"] == parent_id and ext["child"] == child_id:
            quat = np.array([ext["R"]["x"], ext["R"]["y"], ext["R"]["z"], ext["R"]["w"]])
            rot = R.from_quat(quat)
            rot = rot.as_matrix()
            trans = np.array([ext["T"]["x"], ext["T"]["y"], ext["T"]["z"]])
            return np.array(rot.tolist()), trans
    return None, None

def rectify(input_images_path: str = r"fhd_dataset/l/*.png", calibration_json_path: str = "fhd_dataset/fhd_basler_calibration.json",
            cameras_ids: tuple = ("cam1_ir1", "cam1_ir2")) -> np.array:

    # Load intrinsic and extrinsic calibration
    calibration_json = read_custom_json(calibration_json_path)
    # print(calibration_json)
    left_k = read_camera_matrix(calibration_json, cameras_ids[0])
    right_k = read_camera_matrix(calibration_json, cameras_ids[1])
    left_d = read_distortion_vector(calibration_json, cameras_ids[0])
    right_d = read_distortion_vector(calibration_json, cameras_ids[1])
    R, T = read_extrinsic(calibration_json, parent_id=cameras_ids[0], child_id=cameras_ids[1])
    cam1_ir1_images = np.array(sorted(glob.glob(input_images_path)))
    img_left = cv2.imread(cam1_ir1_images[0], cv2.IMREAD_GRAYSCALE)

    # Calculate rectification transform, projection matrix and perspective transformation
    R_left, R_right, P_left, P_right, Q, left_roi, right_roi = cv2.stereoRectify(left_k,
                                                                                 left_d,
                                                                                 right_k,
                                                                                 right_d,
                                                                                 img_left.shape[::-1],
                                                                                 R,
                                                                                 T,
                                                                                 1,
                                                                                 (0, 0),
                                                                                 flags=0,
                                                                                 )
    output_json = {"R": {cameras_ids[0]: R_left.tolist(), cameras_ids[1]: R_right.tolist()},"P": {cameras_ids[0]: P_left.tolist(),cameras_ids[1]: P_right.tolist()}}
    with open(os.path.join('fhd_dataset', 'fhd_basler_rectification.json'), 'w') as f:
        json.dump(output_json, f, indent=4)
    return  left_k,left_d,R_left,P_left, right_k, right_d, R_right, P_right
#         stereo_map_l_x, stereo_map_l_y = cv2.initUndistortRectifyMap(left_k, left_d, R_left, P_left,
#                                                                  img_left.shape[::-1], cv2.CV_16SC2)
#  img_left_rect = cv2.remap(img_left, stereo_map_l_x, stereo_map_l_y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)

def rectify_cv( calibration_json_path: str = "fhd_dataset/fhd_basler_calibration.json",
            cameras_ids: tuple = ("cam1_ir1", "cam1_ir2"),rectification_json_path: str = "fhd_dataset/fhd_basler_rectification.json"):
    cam1_ir1 = r"fhd_dataset/l/*.png"
    cam1_ir2 = r"fhd_dataset/r/*.png"
    cam1_ir1_images = np.array(sorted(glob.glob(cam1_ir1)))
    cam1_ir2_images = np.array(sorted(glob.glob(cam1_ir2)))

    calibration_json = read_custom_json(calibration_json_path)
    left_k = read_camera_matrix(calibration_json, cameras_ids[0])
    right_k = read_camera_matrix(calibration_json, cameras_ids[1])
    left_d = read_distortion_vector(calibration_json, cameras_ids[0])
    right_d = read_distortion_vector(calibration_json, cameras_ids[1])
    rectification = read_custom_json(rectification_json_path)
    R_left = np.array(rectification["R"][cameras_ids[0]])
    R_right = np.array(rectification["R"][cameras_ids[1]])
    P_left = np.array(rectification["P"][cameras_ids[0]])
    P_right = np.array(rectification["P"][cameras_ids[1]])

    for ir1_filename, ir2_filename in zip(cam1_ir1_images, cam1_ir2_images):
        img_left_filename = ir1_filename
        img_right_filename = ir2_filename
        # read files
        img_left = cv2.imread( img_left_filename, cv2.IMREAD_GRAYSCALE)
        img_right = cv2.imread( img_right_filename, cv2.IMREAD_GRAYSCALE)

        # rectify maps
        stereoMapL_x, stereoMapL_y = cv2.initUndistortRectifyMap(left_k, left_d, R_left, P_left, img_left.shape[::-1],
                                                                 cv2.CV_16SC2)
        stereoMapR_x, stereoMapR_y = cv2.initUndistortRectifyMap(right_k, right_d, R_right, P_right,
                                                                 img_right.shape[::-1], cv2.CV_16SC2)

        # executing maps
        img_left_rect = cv2.remap(img_left, stereoMapL_x, stereoMapL_y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
        img_right_rect = cv2.remap(img_right, stereoMapR_x, stereoMapR_y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)


        img_left_filename = img_left_filename.replace("fhd_dataset/l", "fhd_dataset/l_rectified")
        img_right_filename =img_right_filename.replace("fhd_dataset/r", "fhd_dataset/r_rectified")
        if not os.path.exists("fhd_dataset/l_rectified"):
            os.makedirs("fhd_dataset/l_rectified")
        if not os.path.exists("fhd_dataset/r_rectified"):
            os.makedirs("fhd_dataset/r_rectified")
        cv2.imwrite(img_left_filename, img_left_rect)
        cv2.imwrite(img_right_filename, img_right_rect)
if __name__ == "__main__":
    # calibrate()
    # rectify()
    rectify_cv()
