import cv2

stereo = cv2.StereoSGBM_create(minDisparity=0,
                               numDisparities=256,
                               blockSize=5)

wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)

print("getDepthDiscontinuityRadius", wls_filter.getDepthDiscontinuityRadius())
print("getLambda", wls_filter.getLambda())
print("getLRCthresh", wls_filter.getLRCthresh())
print("getROI", wls_filter.getROI())
print("getSigmaColor", wls_filter.getSigmaColor())

