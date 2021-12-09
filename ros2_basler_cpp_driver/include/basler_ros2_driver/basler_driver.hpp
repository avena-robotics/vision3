#ifndef __BASLER_ROS2_DRIVER_HPP__
#define __BASLER_ROS2_DRIVER_HPP__

// __HEADERs__


#include <std_srvs/srv/trigger.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <std_srvs/srv/trigger.hpp>
// _CPP_
#include <chrono>
#include <optional>
#include <filesystem>
#include <shared_mutex>
// _AVENA_
#include "custom_interfaces/srv/get_all_images.hpp"
#include "custom_interfaces/srv/get_color_image.hpp"
#include "custom_interfaces/srv/get_mono_images.hpp"

// _OPENCV_
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/core/affine.hpp>
// _PYLON_
#include "basler_ros2_driver/basler_image_handlers.hpp"
#include "basler_ros2_driver/basler_configuration_handlers.hpp"
// _JSON_

namespace basler
{
    // __Deductions__
    // _PYLON_
    using CamName_CamSerial = std::map<std::string, std::string>;
    // _CPP_
    // _JSON_
    // _AVENA_
    using GetAllImages = custom_interfaces::srv::GetAllImages;
    using GetColorImage = custom_interfaces::srv::GetColorImage;
    using GetMonoImages = custom_interfaces::srv::GetMonoImages;
    using Trigger = std_srvs::srv::Trigger;

    // __Classes__
    class BaslerROS2Driver : public rclcpp::Node
    {
    public:
        // __PUBLIC Member Functions__

        // _AVENA_
        BaslerROS2Driver(const rclcpp::NodeOptions &options);
        BaslerROS2Driver(const BaslerROS2Driver &) = delete;
        BaslerROS2Driver() = delete;
        BaslerROS2Driver &operator=(const BaslerROS2Driver &) = delete;
        virtual ~BaslerROS2Driver();

    private:
        // __ PRIVATE Member VARIABLES__

        // _AVENA_
        // service servers
        rclcpp::Service<GetAllImages>::SharedPtr _get_all_images_server;
        rclcpp::Service<GetColorImage>::SharedPtr _get_color_image_server;
        rclcpp::Service<GetMonoImages>::SharedPtr _get_mono_images_server;
        // _ROS_
        // publishers
        rclcpp::Publisher<Image>::SharedPtr _bgr_color_publisher;
        // timers
        // service servers
        rclcpp::Service<Trigger>::SharedPtr _open_basler_cameras_server;
        rclcpp::Service<Trigger>::SharedPtr _close_basler_cameras_server;
        // _CPP_
        const std::string _color_id{"color"}, _mono_id{"mono"};
        CamName_CamSerial _camera_group_serials{
            {"40093166", "left_" + _mono_id},
            {"40134758", "right_" + _mono_id},
            {"40099899", _color_id}};

        const float _hd_color_fps = 1.0 / 37.0;
        const std::string _mono_postfix = "um";
        const std::string _color_postfix = "uc";
        const uint8_t _request_no_of_cameras = 3;

        // _PYLON_
        std::shared_ptr<Pylon::CInstantCameraArray> _avena_basler_cameras;

        ColorImageEventHandler* _color_handler;
        LeftMonoImageEventHandler* _left_mono_handler;
        RightMonoEventHandler* _right_mono_handler;
        MonoCameraConfigurationHandler* _left_mono_config_handler;
        MonoCameraConfigurationHandler* _right_mono_config_handler;
        ColorCameraConfigurationHandler* _color_config_handler;

        // __PRIVATE Member Functions__
        // _AVENA_
        void _openBaslerCameras(const CamName_CamSerial &camera_group_serials);
        /**
         * @brief closes open camera array
         * @param[in] basler_cameras Array of basler cameras to close
         * \return Error code for success or failure of closing camera devices
         */
        void _closeBaslerCameras();

        void _getAllImagesCb(const std::shared_ptr<GetAllImages::Request> request,
                             std::shared_ptr<GetAllImages::Response> response);
        void _getColorImageCb(const std::shared_ptr<GetColorImage::Request> request,
                              std::shared_ptr<GetColorImage::Response> response);
        void _getMonoImagesCb(const std::shared_ptr<GetMonoImages::Request> request,
                              std::shared_ptr<GetMonoImages::Response> response);
        // _CPP_
        // _ROS_
        void _openBaslerCamerasCb(const std::shared_ptr<Trigger::Request> request,
                                  std::shared_ptr<Trigger::Response> response);
        void _closeBaslerCamerasCb(const std::shared_ptr<Trigger::Request> request,
                                   std::shared_ptr<Trigger::Response> response);
        void _baslerColorHdCb();
        // _PYLON_
        // _JSON_
    };

} // namespace basler

#endif // __BASLER_ROS2_DRIVER_MAIN_HPP__