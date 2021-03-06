#include "basler_ros2_driver/basler_driver.hpp"

namespace basler
{
    BaslerROS2Driver::BaslerROS2Driver(const rclcpp::NodeOptions &options) : Node("basler_ros2_driver", options)
    {
        RCLCPP_INFO(this->get_logger(), "started basler_ros2_driver Node");
        _open_basler_cameras_server = this->create_service<Trigger>("open_basler_cameras", std::bind(&BaslerROS2Driver::_openBaslerCamerasCb, this, std::placeholders::_1, std::placeholders::_2));
        _close_basler_cameras_server = this->create_service<Trigger>("close_basler_cameras", std::bind(&BaslerROS2Driver::_closeBaslerCamerasCb, this, std::placeholders::_1, std::placeholders::_2));
        _get_color_image_server = this->create_service<GetColorImage>("get_basler_color_image", std::bind(&BaslerROS2Driver::_getColorImageCb, this, std::placeholders::_1, std::placeholders::_2));
        _get_mono_images_server = this->create_service<GetMonoImages>("get_basler_mono_images", std::bind(&BaslerROS2Driver::_getMonoImagesCb, this, std::placeholders::_1, std::placeholders::_2));
        _get_all_images_server = this->create_service<GetAllImages>("get_basler_all_images", std::bind(&BaslerROS2Driver::_getAllImagesCb, this, std::placeholders::_1, std::placeholders::_2));
        _bgr_color_publisher = this->create_publisher<Image>("/basler/color/image_raw", rclcpp::SensorDataQoS());
        Pylon::PylonInitialize();
    }
    BaslerROS2Driver::~BaslerROS2Driver()
    {

        if (_avena_basler_cameras != nullptr && (_avena_basler_cameras->IsOpen() || _avena_basler_cameras->IsGrabbing()))
        {
            _closeBaslerCameras();
        }
        rclcpp::shutdown();
    }

    void BaslerROS2Driver::_openBaslerCameras(const CamName_CamSerial &camera_group_serials)
    {
        RCLCPP_INFO(this->get_logger(), "Trying to open cameras");
        try
        {
            Pylon::CTlFactory &tlFactory = Pylon::CTlFactory::GetInstance();
            Pylon::DeviceInfoList_t devices;
            auto no_of_connected_devices = tlFactory.EnumerateDevices(devices);
            if (no_of_connected_devices == 0)
            {
                RCLCPP_ERROR(this->get_logger(), "can not access any cameras");
                return;
            }
            else if (no_of_connected_devices < _request_no_of_cameras)
            {
                RCLCPP_ERROR(this->get_logger(), "can not access all requested cameras");
                return;
            }
            else
            {
                RCLCPP_INFO_STREAM(this->get_logger(), "Number of connected cameras: " << no_of_connected_devices);
                _avena_basler_cameras = std::make_shared<Pylon::CInstantCameraArray>();
                _avena_basler_cameras->Initialize(devices.size());
                for (size_t camera_idx = 0; camera_idx < _avena_basler_cameras->GetSize(); camera_idx++)
                {
                    (*_avena_basler_cameras)[camera_idx].Attach(tlFactory.CreateDevice(devices[camera_idx]));
                    auto device_serial = std::string((*_avena_basler_cameras)[camera_idx].GetDeviceInfo().GetSerialNumber());
                    if (camera_group_serials.find(device_serial) != camera_group_serials.end())
                    {
                        RCLCPP_INFO_STREAM(this->get_logger(), camera_group_serials.at(device_serial) << " camera serial number: " << device_serial);
                        if (camera_group_serials.at(device_serial) == "color")
                        {
                            _color_handler = new ColorImageEventHandler(_bgr_color_publisher);
                            _color_config_handler = new ColorCameraConfigurationHandler();
                            (*_avena_basler_cameras)[camera_idx].RegisterImageEventHandler(_color_handler, Pylon::RegistrationMode_Append, Pylon::Cleanup_Delete);
                            (*_avena_basler_cameras)[camera_idx].RegisterConfiguration(_color_config_handler, Pylon::RegistrationMode_Append, Pylon::Cleanup_Delete);
                        }
                        else if (camera_group_serials.at(device_serial) == "left_mono")
                        {
                            _left_mono_handler = new LeftMonoImageEventHandler();
                            _left_mono_config_handler = new MonoCameraConfigurationHandler();
                            (*_avena_basler_cameras)[camera_idx].RegisterImageEventHandler(_left_mono_handler, Pylon::RegistrationMode_Append, Pylon::Cleanup_Delete);
                            (*_avena_basler_cameras)[camera_idx].RegisterConfiguration(_left_mono_config_handler, Pylon::RegistrationMode_Append, Pylon::Cleanup_Delete);
                        }
                        else if (camera_group_serials.at(device_serial) == "right_mono")
                        {
                            _right_mono_handler = new RightMonoEventHandler();
                            _right_mono_config_handler = new MonoCameraConfigurationHandler();
                            (*_avena_basler_cameras)[camera_idx].RegisterImageEventHandler(_right_mono_handler, Pylon::RegistrationMode_Append, Pylon::Cleanup_Delete);
                            (*_avena_basler_cameras)[camera_idx].RegisterConfiguration(_right_mono_config_handler, Pylon::RegistrationMode_Append, Pylon::Cleanup_Delete);
                        }
                    }
                    else
                    {
                        RCLCPP_WARN(this->get_logger(), "Unsupported Device connected");
                    }
                }
                if (_avena_basler_cameras->IsPylonDeviceAttached())
                {
                    _avena_basler_cameras->StartGrabbing(Pylon::GrabStrategy_LatestImageOnly, Pylon::GrabLoop_ProvidedByInstantCamera);
                    RCLCPP_INFO(this->get_logger(), "cameras are open");
                }
                else
                {
                    RCLCPP_WARN(this->get_logger(), "cameras are not open");
                    return ;
                }
                
            }
        }
        catch (const GenICam_3_1_Basler_pylon::GenericException &e)
        {
            RCLCPP_ERROR(this->get_logger(), e.what());
            return;
        }
    }

    void BaslerROS2Driver::_openBaslerCamerasCb(const std::shared_ptr<Trigger::Request> /*request*/,
                                                std::shared_ptr<Trigger::Response> response)
    {
        RCLCPP_INFO(this->get_logger(), "Open Basler cameras is called");
        if (_avena_basler_cameras == nullptr || _avena_basler_cameras->IsOpen() != true)
        {
            _openBaslerCameras(_camera_group_serials);
            if (_avena_basler_cameras != nullptr && _avena_basler_cameras->IsGrabbing())
            {
                response->success = true;
            }
            else
            {
                response->success = false;
            }
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "cameras are already opened");
            response->success = true;
        }
        if (_avena_basler_cameras != nullptr && _avena_basler_cameras->GetSize() == _camera_group_serials.size())
        {
            response->success = true;
        }
        else
        {
            response->success = false;
        }
    }

    void BaslerROS2Driver::_closeBaslerCameras()
    {
        if (_avena_basler_cameras->IsGrabbing())
        {
            _avena_basler_cameras->StopGrabbing();
        }
        else if (_avena_basler_cameras->IsOpen())
        {
            _avena_basler_cameras->Close();
        }

        for (size_t camera_idx = 0; camera_idx < _avena_basler_cameras->GetSize(); camera_idx++)
        {

            auto device_serial = std::string((*_avena_basler_cameras)[camera_idx].GetDeviceInfo().GetSerialNumber());
            if (_camera_group_serials.find(device_serial) != _camera_group_serials.end())
            {

                if (_camera_group_serials.at(device_serial) == "color")
                {

                    if (_color_handler != nullptr && _color_config_handler != nullptr)
                    {
                        (*_avena_basler_cameras)[camera_idx].DeregisterImageEventHandler(_color_handler);
                        (*_avena_basler_cameras)[camera_idx].DeregisterConfiguration(_color_config_handler);
                    }
                }
                else if (_camera_group_serials.at(device_serial) == "left_mono")
                {

                    if (_left_mono_handler != nullptr && _left_mono_config_handler != nullptr)
                    {

                        (*_avena_basler_cameras)[camera_idx].DeregisterImageEventHandler(_left_mono_handler);
                        (*_avena_basler_cameras)[camera_idx].DeregisterConfiguration(_left_mono_config_handler);
                    }
                }
                else if (_camera_group_serials.at(device_serial) == "right_mono")
                {

                    if (_right_mono_handler != nullptr && _right_mono_config_handler != nullptr)
                    {
                        (*_avena_basler_cameras)[camera_idx].DeregisterImageEventHandler(_right_mono_handler);
                        (*_avena_basler_cameras)[camera_idx].DeregisterConfiguration(_right_mono_config_handler);
                    }
                }
            }
        }

        RCLCPP_INFO(this->get_logger(), "cameras are closed");
    }
    void BaslerROS2Driver::_closeBaslerCamerasCb(const std::shared_ptr<Trigger::Request> /*request*/,
                                                 std::shared_ptr<Trigger::Response> response)
    {
        RCLCPP_INFO(this->get_logger(), "close Basler cameras is called");
        if (_avena_basler_cameras != nullptr && _avena_basler_cameras->GetSize() > 0 && _avena_basler_cameras->IsOpen())
        {
            _closeBaslerCameras();
            response->success = true;
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "cameras are already closed or not open");
            response->success = true;
        }
    }
    void BaslerROS2Driver::_getAllImagesCb(const std::shared_ptr<GetAllImages::Request> /*request*/,
                                           std::shared_ptr<GetAllImages::Response> response)
    {
        RCLCPP_INFO(this->get_logger(), "get all images is called");

        if (_left_mono_handler != nullptr && _right_mono_handler != nullptr && _color_handler != nullptr && _avena_basler_cameras != nullptr && _avena_basler_cameras->IsGrabbing())
        {
            auto left_mono_image = _left_mono_handler->getLeftMonoImage();
            auto right_mono_image = _right_mono_handler->getRightMonoimage();
            auto color_4k_image = _color_handler->get4kColorImage();

            if (left_mono_image.size().width > 0 && right_mono_image.size().width > 0 && color_4k_image.size().width > 0)
            {
                auto left_header = std_msgs::msg::Header();
                auto right_header = std_msgs::msg::Header();
                auto color_header = std_msgs::msg::Header();
                color_header.frame_id = "/basler/color/image_raw";
                left_header.frame_id = "/basler/left/image_raw";
                right_header.frame_id = "/basler/right/image_raw";
                auto now = builtin_interfaces::msg::Time(this->now());
                left_header.stamp = now;
                right_header.stamp = now;
                color_header.stamp = now;
                cv_bridge::CvImagePtr left_cv_ptr = std::make_shared<cv_bridge::CvImage>(left_header, sensor_msgs::image_encodings::MONO8, left_mono_image);
                cv_bridge::CvImagePtr right_cv_ptr = std::make_shared<cv_bridge::CvImage>(right_header, sensor_msgs::image_encodings::MONO8, right_mono_image);
                cv_bridge::CvImagePtr color_cv_ptr = std::make_shared<cv_bridge::CvImage>(color_header, sensor_msgs::image_encodings::RGB8, color_4k_image);

                try
                {
                    response->left_mono = *(left_cv_ptr->toImageMsg());
                    response->right_mono = *(right_cv_ptr->toImageMsg());
                    response->color = *(color_cv_ptr->toImageMsg());
                }
                catch (cv_bridge::Exception &e)
                {
                    RCLCPP_ERROR(this->get_logger(), e.what());
                    return;
                }
            }
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "cameras are not open, return empty response");
            response->left_mono = Image();
            response->right_mono = Image();
            response->color = Image();
        }
    }
    void BaslerROS2Driver::_getColorImageCb(const std::shared_ptr<GetColorImage::Request> /*request*/,
                                            std::shared_ptr<GetColorImage::Response> response)
    {
        RCLCPP_INFO(this->get_logger(), "get Color Image is called");
        if (_color_handler != nullptr && _avena_basler_cameras != nullptr && _avena_basler_cameras->IsGrabbing())
        {
            auto color_4k_image = _color_handler->get4kColorImage();
            if (color_4k_image.size().width > 0)
            {
                auto header = std_msgs::msg::Header();
                header.frame_id = "/basler/color/image_raw";
                header.stamp = builtin_interfaces::msg::Time(this->now());
                cv_bridge::CvImagePtr cv_ptr = std::make_shared<cv_bridge::CvImage>(header, sensor_msgs::image_encodings::RGB8, color_4k_image);
                try
                {
                    response->color = *(cv_ptr->toImageMsg());
                }
                catch (cv_bridge::Exception &e)
                {
                    RCLCPP_ERROR(this->get_logger(), e.what());
                    return;
                }
            }
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "cameras are not open, returning empty response");
            response->color = Image();
        }
    }
    void BaslerROS2Driver::_getMonoImagesCb(const std::shared_ptr<GetMonoImages::Request> /*request*/,
                                            std::shared_ptr<GetMonoImages::Response> response)
    {
        RCLCPP_INFO(this->get_logger(), "get Mono Images is called");
        if (_left_mono_handler != nullptr && _right_mono_handler != nullptr && _avena_basler_cameras != nullptr && _avena_basler_cameras->IsGrabbing())
        {
            auto left_mono_image = _left_mono_handler->getLeftMonoImage();
            auto right_mono_image = _right_mono_handler->getRightMonoimage();
            if (left_mono_image.size().width > 0 && right_mono_image.size().width > 0)
            {
                auto left_header = std_msgs::msg::Header();
                auto right_header = std_msgs::msg::Header();
                left_header.frame_id = "/basler/left/image_raw";
                right_header.frame_id = "/basler/right/image_raw";
                auto now = builtin_interfaces::msg::Time(this->now());
                left_header.stamp = now;
                right_header.stamp = now;
                cv_bridge::CvImagePtr left_cv_ptr = std::make_shared<cv_bridge::CvImage>(left_header, sensor_msgs::image_encodings::MONO8, left_mono_image);
                cv_bridge::CvImagePtr right_cv_ptr = std::make_shared<cv_bridge::CvImage>(right_header, sensor_msgs::image_encodings::MONO8, right_mono_image);
                try
                {
                    response->left_mono = *(left_cv_ptr->toImageMsg());
                    response->right_mono = *(right_cv_ptr->toImageMsg());
                }
                catch (cv_bridge::Exception &e)
                {
                    RCLCPP_ERROR(this->get_logger(), e.what());
                    return;
                }
            }
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "cameras are not open, returning empty response");
            response->left_mono = Image();
            response->right_mono = Image();
        }
    }
} // namespace basler
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(basler::BaslerROS2Driver)