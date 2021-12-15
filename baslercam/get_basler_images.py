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


if __name__ == '__main__':
    BASE_PATH = '/home/avena/datasets'
    CAMERA_1 = os.path.join(BASE_PATH, 'camera_1')
    CAMERA_2 = os.path.join(BASE_PATH, 'camera_2')
    CAMERA_3 = os.path.join(BASE_PATH, 'camera_3')
    os.makedirs(CAMERA_1, exist_ok=True)
    os.makedirs(CAMERA_2, exist_ok=True)
    os.makedirs(CAMERA_3, exist_ok=True)

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
        input('Press ENTER to start grabbing image')
        print('Grabing images')

        start_time = time.perf_counter()
        timeout = 1  # seconds
        cam_images = {i: [] for i in range(cameras.GetSize())}
        while time.perf_counter() - start_time < timeout:
            start_loop = time.perf_counter()
            # Trigger cameras
            for camera in cameras:
                camera.TriggerSoftware.Execute()

            # Read images
            for i, camera in enumerate(cameras):
                grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

                if grab_result and grab_result.GrabSucceeded():
                    cam_images[i].append(grab_result.GetArray())
                else:
                    print(f'Failed retrieving result for {i + 1} camera')
            end_loop = time.perf_counter()
            elapsed_time = end_loop - start_loop
            # TODO: Calculate how much program should wait so that each loop elapses
            # same amount of time.
            time.sleep(0.02)

        print('Done grabbing')

        # Saving images to file
        print('Saving images to files')
        for i in range(len(cam_images[i])):
            ts = time.time_ns()
            for cam_id, images in cam_images.items():
                cv2.imwrite(os.path.join(BASE_PATH, f'camera_{cam_id + 1}', f'{ts}_camera_{cam_id + 1}.png'), images[i])
    
    except genicam.GenericException as e:
        print('[ERROR]:', e)

    cameras.Close()
    print('Cameras closed')

