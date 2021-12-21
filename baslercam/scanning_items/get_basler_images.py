import time
import os
import argparse
import cv2
from queue import Queue
from threading import Thread
from pypylon import pylon
from pypylon import genicam


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
        grab_result.Release()
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
    #########################################################
    # Getting command line arguments from user
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--base_dir', type=str, 
                         help='absolute path to directory for images',
                         default=f'{os.path.join(os.path.expanduser("~"), "dataset")}')
    parser.add_argument('-t', '--time_of_rotation', type=float,
                        help='amount of time (in seconds) one rotation of turntable takes',
                        default=32.508)
    parser.add_argument('-n', '--nr_photos', type=int,
                        help='number of photos to save for each camera',
                        default=36)
    args = parser.parse_args()
    base_path = args.base_dir
    amount_of_photos = args.nr_photos
    time_of_rotation = args.time_of_rotation
    time_between_photos = time_of_rotation / amount_of_photos
    print(f'Saving images to "{base_path}" directory')
    print(f'Amount of photos to take: {amount_of_photos}')
    print(f'One rotation of turntable takes: {time_of_rotation} seconds')
    #########################################################

    try:
        tlfactory= pylon.TlFactory.GetInstance()
        devices = tlfactory.EnumerateDevices()
        if len(devices) == 0:
            raise pylon.RuntimeException("No camera present.")

        cameras: pylon.InstantCameraArray = pylon.InstantCameraArray(len(devices))

        for i, camera in enumerate(cameras):
            device: pylon.DeviceInfo = tlfactory.CreateDevice(devices[i])
            camera.Attach(device)

        cameras.Open()
        for i, camera in enumerate(cameras):
            dir_name = os.path.dirname(os.path.abspath(__file__))
            camera_serial = camera.GetDeviceInfo().GetSerialNumber()
            camera_model = camera_model = camera.GetDeviceInfo().GetModelName()
            config_path = os.path.join(dir_name, 'config', f'{camera_model}_{camera_serial}.pfs')
            pylon.FeaturePersistence.Load(config_path, camera.GetNodeMap(), True)

        print("Cameras are opened and configured")

        input('Press ENTER to start grabbing images')
        print('Grabing images')
        cameras.StartGrabbing(pylon.GrabStrategy_OneByOne, pylon.GrabLoop_ProvidedByUser)

        # Initialize directories for images
        for i in range(cameras.GetSize()):
            os.makedirs(os.path.join(base_path, f'camera_{i + 1}'), exist_ok=True)

        # Initialize saving to file on separate thread
        q = Queue()
        t = Thread(target=save_images, args=(q, base_path))
        t.start()

        start_time = time.perf_counter()
        cnt = 0
        while cnt < amount_of_photos:
            print(f'{cnt + 1} / {amount_of_photos}', end='\r')
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
        print('\nDone grabbing')

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
        
    except (genicam.GenericException, Exception) as e:
        print('[ERROR]:', e)

    cameras.Close()
    print('Cameras closed')
