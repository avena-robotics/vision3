import json
import numpy as np


if __name__ == '__main__':
    # cam0_to_cam3 = np.array(json.load(open("charuco_calibration_cam0_to_cam3.json")))
    # cam5_to_cam3 = np.array(json.load(open("charuco_calibration_cam5_to_cam3.json")))
    # cam3_to_cam2 = np.array(json.load(open("charuco_calibration_cam3_to_cam2.json")))
    # cam2_to_cam4 = np.array(json.load(open("charuco_calibration_cam2_to_cam4.json")))

    # # cam5_to_cam4_chain = cam5_to_cam3 @ cam3_to_cam2 @ cam2_to_cam4
    # cam5_to_cam4_chain = cam2_to_cam4 @ cam3_to_cam2 @ cam5_to_cam3
    # cam0_to_cam4_chain = cam2_to_cam4 @ cam3_to_cam2 @ cam0_to_cam3 
    # cam3_to_cam4_chain = cam2_to_cam4 @ cam3_to_cam2

    # with open("charuco_calibration_cam5_to_cam4.json", 'w') as f:
    #     json.dump(cam5_to_cam4_chain.tolist(), f, indent=3)
    
    # with open("charuco_calibration_cam0_to_cam4.json", 'w') as f:
    #     json.dump(cam0_to_cam4_chain.tolist(), f, indent=3)
    
    # with open("charuco_calibration_cam3_to_cam4.json", 'w') as f:
    #     json.dump(cam3_to_cam4_chain.tolist(), f, indent=3)

    cam0_to_cam2 = np.array(json.load(open("local_calibration_cam0_to_cam2.json")))
    cam2_to_cam3 = np.array(json.load(open("local_calibration_cam2_to_cam3.json")))

    cam0_to_cam3_chain = cam2_to_cam3 @ cam0_to_cam2

    with open("local_calibration_cam0_to_cam3.json", 'w') as f:
        json.dump(cam0_to_cam3_chain.tolist(), f, indent=3)
