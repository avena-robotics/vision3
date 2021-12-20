import time
import os
import cv2
from queue import Queue
from threading import Thread
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


def save_images(q: Queue, base_path: str):
    """
    NOTE: Make sure to create directories for images
    """
    images, done = q.get(block=True)
    while not done:
        ts = time.time_ns()
        for i, image in enumerate(images):
            cv2.imwrite(os.path.join(base_path, f'camera_{i + 1}', f'{ts}_camera_{i + 1}.png'), image)
        q.task_done()
        images, done = q.get(block=True)
    q.task_done()


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

        # Initialize directories for images
        for i in range(cameras.GetSize()):
            os.makedirs(os.path.join(base_path, 'background', f'camera_{i + 1}'), exist_ok=True)
            os.makedirs(os.path.join(base_path, f'camera_{i + 1}'), exist_ok=True)

        # Initialize saving to file on separate thread
        q = Queue()
        t = Thread(target=save_images, args=(q, base_path))
        t.start()

        start_time = time.perf_counter()
        cnt = 0
        while cnt < amount_of_photos:
            start_loop = time.perf_counter()

            # Read images from cameras
            images = get_images(cameras)

            # Send images 
            q.put((images, False))

            cnt += 1
            end_loop = time.perf_counter()
            elapsed_time = end_loop - start_loop

            # Wait remaining time
            time.sleep(time_between_photos - elapsed_time)
        print('Done grabbing')

        # Send request to stop saving images
        q.put((None, True))
        t.join()
        q.join()

        ################################################################
        # Saving background image
        input('Please remove object and press ENTER')
        bg_cam_images = get_images(cameras)
        
        print('Saving background images to files')
        ts = time.time_ns()
        for cam_id, bg_image in enumerate(bg_cam_images):
            dir_path = os.path.join(base_path, 'background', f'camera_{cam_id + 1}')
            os.makedirs(dir_path, exist_ok=True)
            cv2.imwrite(os.path.join(dir_path, f'{ts}_camera_{cam_id + 1}.png'), bg_image)
        
    except genicam.GenericException as e:
        print('[ERROR]:', e)

    cameras.Close()
    print('Cameras closed')
