import cv2
from datetime import datetime

from pypylon import genicam
from pypylon import pylon

from custom_interfaces.srv import GetAllImages, GetMonoImages, GetColorImage

from std_srvs.srv import Trigger
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from rclpy.node import Node
import rclpy

class ColorImageEventHandler(pylon.ImageEventHandler):
    """color image event handler to manage grab result

    It is an event handler to register a callback for color camera grab result.

    Attributes:
        color_image: 4k bgr8 image
        color_hd_image: hd bgr8 image
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.color_image = None
        self.color_hd_image = None

    def OnImageGrabbed(self, camera: pylon.InstantCamera, grab_result: pylon.GrabResult) -> None:
        """Image callback for color image
        
        It is activated each time, the camera grabs an image. 
        We rotate the image for 90 degrees counterclockwise for disparity estimation and resize 4k image to new hd image
        for publishing.
        
        Args:
            camera: instant basler camera object
            grab_result: basler camera grab result
        Returns:
            None
        Raises:
            CVError: An error occurred accessing the cv numpy array.
        """
        if grab_result.GrabSucceeded():
            temp = grab_result.GetArray()
            self.color_image = cv2.rotate(temp, cv2.ROTATE_90_COUNTERCLOCKWISE)
            self.color_hd_image = cv2.resize(self.color_image, (self.color_image.shape[1]//3, self.color_image.shape[0]//3), interpolation = cv2.INTER_AREA)
            grab_result.Release()

class LeftMonoImageEventHandler(pylon.ImageEventHandler):
    """left image event handler to manage grab result

    It is an event handler to register a callback for left mono camera grab result.

    Attributes:
        left_mono_image: fhd mono8 image
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.left_mono_image = None

    def OnImageGrabbed(self, camera: pylon.InstantCamera, grab_result: pylon.GrabResult)->None:
        """Image callback for left mono image
        
        It is activated each time, the camera grabs an image. 
        We rotate the image for 90 degrees counterclockwise for disparity estimation.
        
        Args:
            camera: instant basler camera object
            grab_result: basler camera grab result
        Returns:
            None
        Raises:
            CVError: An error occurred accessing the cv numpy array
        """
        if grab_result.GrabSucceeded():
            temp = grab_result.GetArray()
            self.left_mono_image = cv2.rotate(temp, cv2.ROTATE_90_COUNTERCLOCKWISE)
            grab_result.Release()


class RightMonoEventHandler(pylon.ImageEventHandler):
    """right image event handler to manage grab result

    It is an event handler to register a callback for right mono camera grab result.

    Attributes:
        right_mono_image: fhd mono8 image 
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.right_mono_image = None

    def OnImageGrabbed(self, camera: pylon.InstantCamera, grab_result: pylon.GrabResult)->None:
        """Image callback for mono image
        
        It is activated each time, the camera grabs an image. 
        We rotate the image for 90 degrees counterclockwise for disparity estimation
        
        Args:
            camera: instant basler camera object
            grab_result: basler camera grab result
        Returns:
            None
        Raises:
            CVError: An error occurred accessing the cv numpy array.
        """
        if grab_result.GrabSucceeded():
            temp = grab_result.GetArray()
            self.right_mono_image = cv2.rotate(temp, cv2.ROTATE_90_COUNTERCLOCKWISE)
            grab_result.Release()


class BaslerPyDriver(Node):
    """Basler ROS Node class for python

    It is an basler camera controller, providing services and topics for accessing cameras data

    Attributes:
        qos_profile: publisher qos profile
        camera_group_serials: user cameras serials 
        _camera_group_serials: saved user cameras serials
        _avena_basler_cameras: pypylon Instant cameras array for synchronization
        open_basler_cameras_service_server: ros service server to open cameras
        get_basler_color_image_srv: ros service server to get color image
        get_basler_mono_images_srv: ros service server to get two mono cameras
        get_basler_all_images_srv: ros service server to get 2 mono and 1 bgr images from basler camera
        basler_camera_color_pub: ros publisher to publish hd color image for security
        color_fps: timer frequency for publishing hd color image
        timer: timer ros object to publish image in timed callback
        bridge: converter of opencv to and from ros images
        _color_image_event_handler: image event handler for color camera
        _left_mono_image_event_handler: image event handler for left mono camera
        _right_mono_event_handler: image event handler for right mono camera

    Example:

        rclpy.init()
        camera_group_serials = {'left_mono': '40132642', 'color': '40120189', 'right_mono': '40134745'}
        basler_py_driver_node = BaslerPyDriver(qos_profile_sensor_data,camera_group_serials)
        try:
            while rclpy.ok():
                rclpy.spin_once(basler_py_driver_node)
                if basler_py_driver_node._avena_basler_cameras is not None and basler_py_driver_node._avena_basler_cameras.IsOpen() and
                not basler_py_driver_node._avena_basler_cameras.IsGrabbing():
                    basler_py_driver_node.get_logger().warn("cameras are not grabbing")
        except KeyboardInterrupt:
            basler_py_driver_node.get_logger().warn("exit key is pressed")
            if basler_py_driver_node._avena_basler_cameras is not None and basler_py_driver_node._avena_basler_cameras.GetSize() > 0:
                basler_py_driver_node.close_basler_cameras(basler_py_driver_node._avena_basler_cameras)
        except ExternalShutdownException:
            basler_py_driver_node.get_logger().warn("external shutdown")
            if basler_py_driver_node._avena_basler_cameras is not None and basler_py_driver_node._avena_basler_cameras.GetSize() > 0:
                basler_py_driver_node.close_basler_cameras(basler_py_driver_node._avena_basler_cameras)
        finally:
            rclpy.try_shutdown()
            basler_py_driver_node.destroy_node()

    """
    def __init__(self, qos_profile:rclpy.qos.QoSProfile,camera_group_serials: dict) -> None:
        super().__init__('basler_camera_controller')
        self.get_logger().info("BaslerPyDriver initialized")
        self._camera_group_serials = camera_group_serials
        self._avena_basler_cameras = None
        self.open_basler_cameras_service_server = self.create_service(Trigger, 'open_basler_cameras', self.open_basler_cameras_cb)
        self.close_basler_cameras_srv = self.create_service(Trigger, 'close_basler_cameras', self.close_basler_cameras_cb)
        self.get_basler_color_image_srv = self.create_service(GetColorImage, 'get_basler_color_image', self.get_basler_color_image_cb)
        self.get_basler_mono_images_srv = self.create_service(GetMonoImages, 'get_basler_mono_images', self.get_basler_mono_images_cb)
        self.get_basler_all_images_srv = self.create_service(GetAllImages, 'get_basler_all_images', self.get_basler_all_images_cb)
        self.basler_camera_color_pub = self.create_publisher(Image, '/basler/color/image_raw', qos_profile)
        self.color_fps = 1.0/37.0 # seconds
        self.timer = None
        self.bridge = CvBridge()
        self._color_image_event_handler = None
        self._left_mono_image_event_handler = None
        self._right_mono_event_handler = None

    def open_basler_cameras(self,camera_group_serials: dict)->pylon.InstantCameraArray:
        """It is responsible for opening multicamera and start grabbing data 
        
        It registers image event handlers for cameras as callback, creates timer
        for publishing color image for security

        Args:
            camera_group_serials: serial numbers for cameras to accquire data
        Returns:
            pylon array of cameras
        Raises:
            GeniCamError: An error occurred accessing busy cameras
        """
        self.get_logger().info("Trying to open cameras")
        self.timer = self.create_timer(self.color_fps, self.basler_color_hd_callback)
        try:
            tlfactory= pylon.TlFactory.GetInstance()
            devices = tlfactory.EnumerateDevices()
            if len(devices) == 0:
                raise pylon.RuntimeException("No camera present.")
            cameras:pylon.InstantCameraArray = pylon.InstantCameraArray(len(devices))
            for i, camera in enumerate(cameras):
                device: pylon.DeviceInfo = tlfactory.CreateDevice(devices[i])
                camera.Attach(device)
                camera_serial = camera.GetDeviceInfo().GetSerialNumber()
                for basler_camera_name , basler_serial_number in camera_group_serials.items():
                    if basler_serial_number == camera_serial and basler_camera_name == 'color':
                        self._color_image_event_handler =ColorImageEventHandler() 
                        camera.RegisterImageEventHandler(self._color_image_event_handler, pylon.RegistrationMode_Append, pylon.Cleanup_Delete)
                    if basler_serial_number == camera_serial and basler_camera_name == 'left_mono':
                        self._left_mono_image_event_handler = LeftMonoImageEventHandler()
                        camera.RegisterImageEventHandler(self._left_mono_image_event_handler, pylon.RegistrationMode_Append, pylon.Cleanup_Delete)
                    if basler_serial_number == camera_serial and basler_camera_name == 'right_mono':
                        self._right_mono_event_handler = RightMonoEventHandler()
                        camera.RegisterImageEventHandler(self._right_mono_event_handler, pylon.RegistrationMode_Append, pylon.Cleanup_Delete)
            cameras.Open()
            for camera in cameras:
                camera = self.configure_basler_camera(camera)
            cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly,pylon.GrabLoop_ProvidedByInstantCamera)
            self.get_logger().info("cameras are open")
            return cameras
        except genicam.GenericException as e:
            self.get_logger().error(e.GetDescription())
            self.get_logger().error("cameras failed to open")
            return None

    def open_basler_cameras_cb(self, request: Trigger.Request , response: Trigger.Response) -> Trigger.Response:
        """Callback for opening cameras as a response to service call

        Args:
            request: empty call
            response: done or failed
        Returns:
            succes or failure
        Raises:
            None
        """
        self.get_logger().info("Open Basler cameras is called")
        if self._avena_basler_cameras is None:
            self._avena_basler_cameras = self.open_basler_cameras(self._camera_group_serials)
        else:
            self.get_logger().warn("cameras are already opened, Stop messing around")
        if self._avena_basler_cameras is not None and self._avena_basler_cameras.GetSize() == len(self._camera_group_serials):
            response.success = True
        else:
            response.success = False
        return response

    def close_basler_cameras(self, cameras:pylon.InstantCameraArray) -> None:
        """close cameras and stop grabbing if cameras are open or grabbing

        Args:
            cameras: basler cameras to close 
        Returns:
            None
        Raises:
            None
        """
        if cameras.IsGrabbing():
            cameras.StopGrabbing()
        if cameras.IsOpen():
            cameras.Close()
        self.get_logger().info("cameras are closed")

    def close_basler_cameras_cb(self, request: Trigger.Request , response: Trigger.Response) -> Trigger.Response:
        """Callback for closing cameras as a response to service call

        Args:
            request: empty call
            response: done or failed
        Returns:
            succes or failure
        Raises:
            None
        """
        self.get_logger().info("close Basler cameras is called")
        if self._avena_basler_cameras is not None and self._avena_basler_cameras.GetSize() > 0:
            self.close_basler_cameras(self._avena_basler_cameras)
            self._avena_basler_cameras = None
            self.timer.cancel()
            response.success = True
        else:
            self.get_logger().warn("cameras are already closed or not open")
            response.success = True
        return response


    def get_basler_color_image_cb(self, request:GetColorImage.Request, response: GetColorImage.Response) -> GetColorImage.Response:
        """Callback for geting color image from camera as a response to service call

        Args:
            request: empty call
            response: color image
        Returns:
            color image 
        Raises:
            None
        """
        self.get_logger().info("get_basler_color_image is called")
        if self._color_image_event_handler.color_image is not None:
            response.color = self.bridge.cv2_to_imgmsg(self._color_image_event_handler.color_image, "bgr8") 
        return response

    def get_basler_mono_images_cb(self, request: GetMonoImages.Request, response: GetMonoImages.Response) -> GetMonoImages.Response:
        """Callback for geting mono images from cameras as a response to service call

        Args:
            request: empty call
            response: mono images
        Returns:
            two mono images
        Raises:
            None
        """
        self.get_logger().info("get_basler_mono_images is called")
        if self._left_mono_image_event_handler.left_mono_image is not None:
            response.left_mono =  self.bridge.cv2_to_imgmsg(self._left_mono_image_event_handler.left_mono_image, "mono8") 
        if self._right_mono_event_handler.right_mono_image is not None:
            response.right_mono  =  self.bridge.cv2_to_imgmsg(self._right_mono_event_handler.right_mono_image, "mono8")  
        return response

    def get_basler_all_images_cb(self, request: GetAllImages.Response, response: GetAllImages.Response) -> GetAllImages.Response:
        """Callback for geting all images from cameras as a response to service call

        Args:
            request: empty call
            response: mono and color images
        Returns:
            two mono images and one bgr image
        Raises:
            None
        """
        self.get_logger().info("get_basler_all_images is called")
        if self._color_image_event_handler.color_image is not None:
            response.color = self.bridge.cv2_to_imgmsg(self._color_image_event_handler.color_image, "bgr8") 
        if self._left_mono_image_event_handler.left_mono_image is not None:
            response.left_mono =  self.bridge.cv2_to_imgmsg(self._left_mono_image_event_handler.left_mono_image, "mono8") 
        if self._right_mono_event_handler.right_mono_image is not None:
            response.right_mono  =  self.bridge.cv2_to_imgmsg(self._right_mono_event_handler.right_mono_image, "mono8") 
        return response

    def basler_color_hd_callback(self) -> None:
        """Callback for publishing hd color image for security
        Args:
            None
        Returns:
            None
        Raises:
            None
        """
        if self._color_image_event_handler is not None and self._color_image_event_handler.color_hd_image is not None:
            hd_color_image = self.bridge.cv2_to_imgmsg(self._color_image_event_handler.color_hd_image, "bgr8") 
            now = datetime.now()
            hd_color_image.header.frame_id = "/basler/color"
            hd_color_image.header.stamp.sec =now.second
            hd_color_image.header.stamp.nanosec =now.microsecond//1000
            if self.basler_camera_color_pub.get_subscription_count() > 0 and hd_color_image is not None:
                # Very Slow (average 55 ms) TODO: FIX https://github.com/ros2/rclpy/issues/856
                self.basler_camera_color_pub.publish(hd_color_image)

    def configure_basler_camera(self,camera: pylon.InstantCamera)->pylon.InstantCamera:
        """Configure camera for auto exposure, binning, pixel format and fps
        Args:
            camera: basler camera object
        Returns:
            camera: basler camera object after configuration
        Raises:
            None
        """
        ColorAutoTargetBrightness=0.15
        MonoAutoTargetBrightness=0.2
        MonoBinning=1
        # 18.7 fps x 3840 x 2160 x 1 byte ~= 160000000 bytes/s
        camera.DeviceLinkThroughputLimit.SetValue(320000000) # 37 fps
        # 37f ps x 3840 x 2160 x 1 byte ~= 320000000 byte/s (mono-8)  (3842241000 - 45 fps) # https://docs.baslerweb.com/exposure-time.html
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
        
        camera_model = camera.GetDeviceInfo().GetModelName()
        mono_postfix = "um"
        color_postfix = "uc"

        if mono_postfix in camera_model:
            if camera.BinningVertical.GetValue() != MonoBinning:
                camera.BinningVertical.SetValue(MonoBinning)

            if camera.BinningHorizontal.GetValue() != MonoBinning:
                camera.BinningHorizontal.SetValue(MonoBinning)

            if camera.BinningHorizontalMode.GetValue() != "Average":
                camera.BinningHorizontalMode.SetValue("Average")

            if camera.BinningVerticalMode.GetValue() != "Average":
                camera.BinningVerticalMode.SetValue("Average")

            if camera.AutoTargetBrightness.GetValue() != MonoAutoTargetBrightness:
                camera.AutoTargetBrightness.SetValue(MonoAutoTargetBrightness)

        if color_postfix in camera_model:
            if camera.AutoTargetBrightness.GetValue() != ColorAutoTargetBrightness:
                camera.AutoTargetBrightness.SetValue(ColorAutoTargetBrightness)
            camera.PixelFormat = 'BGR8'

        return camera
