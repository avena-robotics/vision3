/**
 * This file is part of basler_ros2_driver ROS2 Package
 * License declaration
 * @file  basler_driver
 * @name basler_driver.hpp
 * @addtogroup basler_ros2_driver
 * @image diagram data/camera_controller.png
 * @author Hasan Farag
 * @see [3d-processing](https://app.developerhub.io/robotics/v3.0/robotics/3d-processing)
 * @ref See also pylon docs for daa3840-45um and daa3840-45uc
 */
#ifndef __BASLER_ROS2_DRIVER_HPP__
#define __BASLER_ROS2_DRIVER_HPP__
/**
 *  @defgroup Headers file headers
 */
/** @defgroup ROS2 header files for ROS2
 * @ingroup headers
 *  @{
 */
#include <std_srvs/srv/trigger.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <std_srvs/srv/trigger.hpp>
/** @} */ // end of ROS2

/** @defgroup Cpp  header files for cpp
 * @ingroup headers
 *  @{
 */
#include <chrono>
#include <optional>
#include <filesystem>
/** @} */ // end of Cpp

/** @defgroup Avena  header files for avena
 * @brief custom ros2 interfaces
 * @ingroup headers
 *  @{
 */
#include "custom_interfaces/srv/get_all_images.hpp"
#include "custom_interfaces/srv/get_color_image.hpp"
#include "custom_interfaces/srv/get_mono_images.hpp"
/** @} */ // end of Avena

/** @defgroup Opencv header files for opencv
 * @ingroup headers
 *  @{
 */
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/core/affine.hpp>
/** @} */ // end of Opencv

/** @defgroup Pylon header files for pylon
 * @ingroup headers
 *  @{
 */
#include "basler_ros2_driver/basler_image_handlers.hpp"
#include "basler_ros2_driver/basler_configuration_handlers.hpp"
/** @} */ // end of Pylon

/** @defgroup basler The basler namespace
 *  @brief wraps all pylon driver of dart usb3 basler cameras in ros2 functionalities
 */
namespace basler
{
    /** @defgroup deductions type deductions
     *  @{
     */
    using CamName_CamSerial = std::map<std::string, std::string>;
    using GetAllImages = custom_interfaces::srv::GetAllImages;
    using GetColorImage = custom_interfaces::srv::GetColorImage;
    using GetMonoImages = custom_interfaces::srv::GetMonoImages;
    using Trigger = std_srvs::srv::Trigger;
    /** @} */ // end of deductions

    /**
     * @class BaslerROS2Driver  main ROS2 node class
     * @ingroup basler
     * @brief  represents ROS2 node for basler driver
     */
    class BaslerROS2Driver : public rclcpp::Node
    {
    public:
        /**
         * @brief Construct a new Basler ROS2 Driver object
         * @param[in] options  ros2 node options
         */
        BaslerROS2Driver(const rclcpp::NodeOptions &options);
        /**
         * @brief Construct a new Basler ROS2 Driver object
         */
        BaslerROS2Driver(const BaslerROS2Driver &) = delete;
        /**
         * @brief Construct a new Basler ROS2 Driver object
         */
        BaslerROS2Driver() = delete;
        /**
         * @brief assign a new Basler ROS2 Driver object
         * @return BaslerROS2Driver&
         */
        BaslerROS2Driver &operator=(const BaslerROS2Driver &) = delete;
        /**
         * @brief Destroy the Basler ROS2 Driver object
         */
        virtual ~BaslerROS2Driver();

    private:
        ///@{
        /** Basler images servers */
        rclcpp::Service<GetAllImages>::SharedPtr _get_all_images_server;
        rclcpp::Service<GetColorImage>::SharedPtr _get_color_image_server;
        rclcpp::Service<GetMonoImages>::SharedPtr _get_mono_images_server;
        ///@}
        /**
         * @brief security color image publisher
         */
        rclcpp::Publisher<Image>::SharedPtr _bgr_color_publisher;
        ///@{
        /** Basler open and close services */
        rclcpp::Service<Trigger>::SharedPtr _open_basler_cameras_server;
        rclcpp::Service<Trigger>::SharedPtr _close_basler_cameras_server;
        ///@}
        /**
         * @brief camera basler type ids
         * @showinitializer
         */
        const std::string _color_id{"color"}, _mono_id{"mono"};
        /**
         * @brief camera serials dart usb3 serials
         * @showinitializer
         */
        CamName_CamSerial _camera_group_serials{
            {"40093166", "left_" + _mono_id},
            {"40134758", "right_" + _mono_id},
            {"40099899", _color_id}};

        /**
         * @brief Maximum number of simultaneous cameras supported
         * @showinitializer
         */
        const uint8_t _request_no_of_cameras = 3;

        /**
         * @brief pylon objects for cameras
         */
        std::shared_ptr<Pylon::CInstantCameraArray> _avena_basler_cameras;
        ///@{
        /** Pylon Image and configuration Handlers */
        ColorImageEventHandler *_color_handler;
        LeftMonoImageEventHandler *_left_mono_handler;
        RightMonoEventHandler *_right_mono_handler;
        MonoCameraConfigurationHandler *_left_mono_config_handler;
        MonoCameraConfigurationHandler *_right_mono_config_handler;
        ColorCameraConfigurationHandler *_color_config_handler;
        ///@}
        /**
         * @brief open basler camera array
         * @param[in] camera_group_serials  basler usb3 cameras to stream for ros services and topics
         * For Example: {{"40093166", "left_mono" }, {"40134758", "right_mono"}, {"40099899", "color"}}
         * @throws GeniException
         */
        void _openBaslerCameras(const CamName_CamSerial &camera_group_serials);

        /**
         * @brief closes active basler camera array
         * \return none
         */

        void _closeBaslerCameras();
        /**
         * @brief
         *
         * @param[in] request empty request
         * @param[out] response all mono and color images
         */
        void _getAllImagesCb(const std::shared_ptr<GetAllImages::Request> request,
                             std::shared_ptr<GetAllImages::Response> response);

        /**
         * @brief
         *
         * @param[in] request empty request
         * @param[out] response 4k color bgr image
         */
        void _getColorImageCb(const std::shared_ptr<GetColorImage::Request> request,
                              std::shared_ptr<GetColorImage::Response> response);
        /**
         * @brief
         *
         * @param[in] request empty request
         * @param[out] response right and left 4k mono images
         */
        void _getMonoImagesCb(const std::shared_ptr<GetMonoImages::Request> request,
                              std::shared_ptr<GetMonoImages::Response> response);
        // _CPP_
        // _ROS_
        /**
         * @brief
         *
         * @param[in] request empty trigger request
         * @param[out] response success or failure
         */
        void _openBaslerCamerasCb(const std::shared_ptr<Trigger::Request> request,
                                  std::shared_ptr<Trigger::Response> response);
        /**
         * @brief
         *
         * @param[in] request empty trigger request
         * @param[out] response success or failure
         */
        void _closeBaslerCamerasCb(const std::shared_ptr<Trigger::Request> request,
                                   std::shared_ptr<Trigger::Response> response);
    };

} // namespace basler
/** @} */ // end of group

#endif // __BASLER_ROS2_DRIVER_MAIN_HPP__