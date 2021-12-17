import time
import os
import cv2
from pypylon import pylon
from pypylon import genicam


EXPOSURE_TIME_US = 5000
EXPOSURE_TIME_MS = int(EXPOSURE_TIME_US / 1000)


def configure_camera(camera: pylon.InstantCamera):
    camera.DeviceLinkThroughputLimit.SetValue(320000000) # 37 fps

    # print('\n'.join(dir(camera)))
    # Set 4K resolution
    camera.Width.SetValue(3840)
    camera.Height.SetValue(2160)
    camera.OffsetX.SetValue(0)
    camera.OffsetY.SetValue(0)

    # ROI
    camera.AutoFunctionROISelector.SetValue("ROI1")
    camera.AutoFunctionROIOffsetX.SetValue(0)
    camera.AutoFunctionROIOffsetY.SetValue(0)
    camera.AutoFunctionROIWidth.SetValue(camera.Width.GetValue())
    camera.AutoFunctionROIHeight.SetValue(camera.Height.GetValue())
    camera.AutoFunctionROIUseBrightness.SetValue(False)
    camera.AutoFunctionROIUseWhiteBalance.SetValue(False)

    camera.AutoFunctionROISelector.SetValue("ROI2")
    camera.AutoFunctionROIOffsetX.SetValue(0)
    camera.AutoFunctionROIOffsetY.SetValue(0)
    camera.AutoFunctionROIWidth.SetValue(camera.Width.GetValue())
    camera.AutoFunctionROIHeight.SetValue(camera.Height.GetValue())
    camera.AutoFunctionROIUseBrightness.SetValue(False)
    camera.AutoFunctionROIUseWhiteBalance.SetValue(False)

    # Turn off all auto functions
    camera.BalanceWhiteAuto.SetValue('Off')
    camera.GainAuto.SetValue('Off')
    camera.ExposureMode.SetValue('Timed')
    camera.ExposureAuto.SetValue('Off')

    # Gain value to minimum possible to not add noise
    camera.Gain.SetValue(0)

    # Set exposure time in microseconds
    camera.ExposureTime.SetValue(EXPOSURE_TIME_US)

    # Set pixel format for color
    camera.PixelFormat.SetValue('BGR8')

    # Configure triggers
    camera.TriggerSelector.SetValue('FrameStart')
    camera.TriggerMode.SetValue('On')
    camera.TriggerSource.SetValue('Software')

    # camera.RegisterConfiguration(pylon.SoftwareTriggerConfiguration(), 
    #                              pylon.RegistrationMode_ReplaceAll,
    #                              pylon.Cleanup_Delete)


def get_images(cameras: pylon.InstantCameraArray):
    # Trigger cameras
    for camera in cameras:
        camera.TriggerSoftware.Execute()

    # Read images
    images = [None] * cameras.GetSize()
    for i, camera in enumerate(cameras):
        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grab_result and grab_result.GrabSucceeded():
            images[i] = grab_result.GetArray()
        else:
            print(f'Failed retrieving result for {i + 1} camera')
    return images


if __name__ == '__main__':
    ########################################################
    # User configuration
    base_path = '/home/avena/basler/dataset'
    amount_of_photos = 37
    time_between_photos = 0.903 # seconds
    ########################################################

    try:
        tlfactory= pylon.TlFactory.GetInstance()
        devices = tlfactory.EnumerateDevices()
        if len(devices) == 0:
            raise pylon.RuntimeException("No camera present.")

        cameras: pylon.InstantCameraArray = pylon.InstantCameraArray(len(devices))

        for i, camera in enumerate(cameras):
            device: pylon.DeviceInfo = tlfactory.CreateDevice(devices[i])
            camera.Attach(device)
            print("Using device:", camera.GetDeviceInfo().GetModelName())

        cameras.Open()
        for i, camera in enumerate(cameras):
            configure_camera(camera)
        print("Cameras are opened and configured")

        cameras.StartGrabbing(pylon.GrabStrategy_OneByOne, pylon.GrabLoop_ProvidedByUser)
        input('Press ENTER to start grabbing images')
        print('Grabing images')

        start_time = time.perf_counter()
        cam_images = [None] * amount_of_photos
        times = [None] * amount_of_photos
        cnt = 0
        while cnt < amount_of_photos:
            start_loop = time.perf_counter()

            # Read images from cameras
            cam_images[cnt] = get_images(cameras)

            times[cnt] = time.time_ns()
            end_loop = time.perf_counter()
            cnt += 1
            elapsed_time = end_loop - start_loop

            # Wait remaining time
            time.sleep(time_between_photos - elapsed_time)
        print('Done grabbing')

        ################################################################
        # Saving background image
        input('Please remove object from turntable and press ENTER')
        bg_cam_images = get_images(cameras)
        
        print('Saving background images to files')
        os.makedirs(os.path.join(base_path, 'background', 'camera_1'), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'background', 'camera_2'), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'background', 'camera_3'), exist_ok=True)
        ts = time.time_ns()
        for cam_id, bg_image in enumerate(bg_cam_images):
            cv2.imwrite(os.path.join(base_path, 'background', f'camera_{cam_id + 1}', f'{ts}_camera_{cam_id + 1}.png'), bg_image)

        ################################################################
        # Saving images to file
        print('Saving images to files')
        os.makedirs(os.path.join(base_path, 'camera_1'), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'camera_2'), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'camera_3'), exist_ok=True)
        for i, images in enumerate(cam_images):
            for cam_id in range(cameras.GetSize()):
                cv2.imwrite(os.path.join(base_path, f'camera_{cam_id + 1}', f'{times[i]}_camera_{cam_id + 1}.png'), images[cam_id])
            
    except genicam.GenericException as e:
        print('[ERROR]:', e)

    cameras.Close()
    print('Cameras closed')

