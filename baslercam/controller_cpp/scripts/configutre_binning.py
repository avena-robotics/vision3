# #### 0. Imports
import sys

import cv2
from pypylon import pylon
import os
from datetime import datetime
import time


# ### open_basler_camera function is responsible for opening single camera
# The function is prepared for Basler cameras.
#
# #### Input:
# * serial_number - string
#
# #### Output:
# * camera - pypylon object

def open_basler_camera(serial_number):
    dev_info = None
    for dev in pylon.TlFactory.GetInstance().EnumerateDevices():
        if dev.GetSerialNumber() == serial_number:
            dev_info = dev
            break
    else:
        print('Camera with {} serial number not found'.format(serial_number))
    if dev_info is not None:
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(dev_info))
        if not camera.IsOpen():
            camera.Open()
        return camera
    else:
        return None


# ### configure_basler_camera function is responsible for configure single camera
# The function is prepared for Basler cameras auto settings.
#
# #### Input:
# * camera - pypylon object
# * ColorAutoTargetBrightness - float
# * MonoAutoTargetBrightness - float
#
# #### Output:
# * camera - pypylon object

def configure_basler_camera(camera, ColorAutoTargetBrightness=0.15, MonoAutoTargetBrightness=0.20):
    camera.AutoFunctionROISelector.SetValue("ROI1")
    camera.AutoFunctionROIOffsetX.SetValue(0)
    camera.AutoFunctionROIOffsetY.SetValue(0)
    camera.AutoFunctionROIWidth.SetValue(camera.Width.GetValue())
    camera.AutoFunctionROIHeight.SetValue(camera.Height.GetValue())
    camera.AutoFunctionROIUseBrightness.SetValue(True)

    if camera.ExposureAuto.GetValue() != "Continuous":
        camera.ExposureAuto.SetValue("Continuous")

    if camera.GainAuto.GetValue() != "Continuous":
        camera.GainAuto.SetValue("Continuous")

    mono_postfix = "um"
    if mono_postfix in camera.GetDeviceInfo().GetModelName():
        if camera.BinningVertical.GetValue() != 2:
            camera.BinningVertical.SetValue(2)

        if camera.BinningHorizontal.GetValue() != 2:
            camera.BinningHorizontal.SetValue(2)

        if camera.BinningHorizontalMode.GetValue() != "Sum":
            camera.BinningHorizontalMode.SetValue("Sum")

        if camera.BinningVerticalMode.GetValue() != "Sum":
            camera.BinningVerticalMode.SetValue("Sum")

        if camera.AutoTargetBrightness.GetValue() != MonoAutoTargetBrightness:
            camera.AutoTargetBrightness.SetValue(MonoAutoTargetBrightness)

    color_postfix = "uc"
    if color_postfix in camera.GetDeviceInfo().GetModelName():
        if camera.AutoTargetBrightness.GetValue() != ColorAutoTargetBrightness:
            camera.AutoTargetBrightness.SetValue(ColorAutoTargetBrightness)

    return camera


# ###  get_basler_camera_image function is responsible for getting single image from camera
# The function grabs one image from input camera whether mono or color sensor and saves it to output folder
#
# #### Input:
# * camera - pypylon object
# * output_folder - string
#
# #### Output:
# * None

def get_basler_camera_image(camera, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if camera is None or not camera.IsOpen():
        raise ValueError("Camera object {} is closed.".format(camera))

    converter = pylon.ImageFormatConverter()
    image_type = None
    mono_postfix = "um"
    color_postfix = "uc"
    if color_postfix in camera.GetDeviceInfo().GetModelName():
        image_type = "color"
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    elif mono_postfix in camera.GetDeviceInfo().GetModelName():
        image_type = "mono"
        converter.OutputPixelFormat = pylon.PixelType_Mono8
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    grab_result = camera.GrabOne(1000)
    now = datetime.now()
    timestr = now.strftime("%Y%m%d-%H%M%S")
    serial_number = camera.GetDeviceInfo().GetSerialNumber()
    image = converter.Convert(grab_result)
    img = image.GetArray()
    grab_result.Release()
    camera_id = camera_group_ids[serial_number]
    camera_folder = output_folder +"/" + camera_id
    if not os.path.exists(camera_folder):
        os.makedirs(camera_folder)
    # cv2.imwrite(output_folder+ timestr +f"-{serial_number}-{image_type}-image.png",img)
    # image_metadata = output_folder + timestr + f"-{serial_number}-{image_type}-image.png"
    image_metadata = camera_folder
    return {image_metadata: img}


# ###  close_basler_camera function is responsible for close camera device
# The function is prepared for Basler cameras.
#
# #### Input:
# * camera - pypylon object
#
# #### Output:
# * None

def close_basler_camera(camera):
    if camera.IsOpen():
        camera.Close()


# ###  open_camera_group function is responsible for open and configure multi-camera avena device
# The function is prepared for Basler cameras.
#
# #### Input:
# * avena_camera_serials - list of strings
#
# #### Output:
# * avena_basler_cameras - pypylon objects


def open_camera_group(avena_camera_serials):
    avena_basler_cameras = []
    for basler_camera_serial in avena_camera_serials:
        basler_camera = open_basler_camera(basler_camera_serial)
        basler_camera = configure_basler_camera(basler_camera, ColorAutoTargetBrightness=0.2,
                                                MonoAutoTargetBrightness=0.25)
        avena_basler_cameras.append(basler_camera)
    return avena_basler_cameras


# ###  get_camera_group_images function is responsible for getting 3 images (mono.color,mono) from avena camera
# The function gets 3 images from avena camera and saves them to output folder
#
# #### Input:
# * avena_basler_cameras - pypylon objects
# * output_folder - string
#
# #### Output:
# * images


def get_camera_group_images(basler_cameras, output_folder):
    images = []
    for basler_camera in basler_cameras:
        images.append(get_basler_camera_image(basler_camera, output_folder))
    return images


# ###  close_camera_group function is responsible for close avena camera devices
# The function is prepared for Basler cameras.
#
# #### Input:
# * basler_cameras - pypylon objects
#
# #### Output:
# * None


def close_camera_group(basler_cameras):
    for basler_camera in basler_cameras:
        close_basler_camera(basler_camera)


# # PIPELINE

# #### 1. Define camera serials

camera_group_serials = ['40093166', '40099899', '40134758']
camera_group_ids ={'40093166':'l', '40099899':'color', '40134758':'r'}

# #### 2. Define output folder


output_folder = "output/"

# #### 3. Open avena camera

avena_camera = open_camera_group(camera_group_serials)
time.sleep(1)
print("cameras opened")
print("cameras binning 2 is set")
close_camera_group(avena_camera)
print("cameras closed")
sys.exit()

