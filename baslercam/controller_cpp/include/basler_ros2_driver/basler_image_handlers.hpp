/**
 * This file is part of basler_ros2_driver ROS2 Package
 * License declaration
 * @file  basler_image_handlers
 * @name basler_image_handlers.hpp
 * @addtogroup basler_ros2_driver
 * @author Hasan Farag
 * @link [3d-processing] https://app.developerhub.io/robotics/v3.0/robotics/3d-processing
 * @ref See also pylon docs for daa3840-45um and daa3840-45uc
 */
#ifndef __BASLER_IMAGE_HANDLER_HPP__
#define __BASLER_IMAGE_HANDLER_HPP__
/** @defgroup Pylon header files for Pylon
 * @ingroup headers
 *  @{
 */
#include <pylon/PylonIncludes.h>
/** @} */ // end of Pylon

/** @defgroup Opencv header files for opencv
 * @ingroup headers
 *  @{
 */
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
/** @} */ // end of Opencv

/** @defgroup ROS2 header files for ROS2
 * @ingroup headers
 *  @{
 */
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
/** @} */ // end of ROS2

/** @defgroup basler The basler namespace
 *  @brief wraps all pylon driver of dart usb3 basler cameras in ros2 functionalities
 */
namespace basler
{
    /** @defgroup deductions type deductions
     *  @{
     */
    using Image = sensor_msgs::msg::Image;
    /** @} */ // end of deductions

    /**
     * @class ColorImageEventHandler to grab color image on callback
     * @ingroup basler
     * @brief called to grab color, rotate and reisze 4k image
     *
     */
    class ColorImageEventHandler : public Pylon::CImageEventHandler
    {
    public:
        /**
         * @brief Construct a new Color Image Event Handler object
         * @param[in] hd_image_publihser ros2 publsiher for security hd image
         */
        explicit ColorImageEventHandler(const rclcpp::Publisher<Image>::SharedPtr &hd_image_publihser);
        /**
         * @brief Construct a new Color Image Event Handler object
         */
        ColorImageEventHandler(const ColorImageEventHandler &) = delete;
        /**
         * @brief used to register color image event handler
         * @return ColorImageEventHandler& color camera event handler
         */
        ColorImageEventHandler &operator=(const ColorImageEventHandler &) = delete;
        /**
         * @brief Destroy the Color Image Event Handler object
         */
        virtual ~ColorImageEventHandler() = default;
        /**
         * @brief image grab callback is used to read color image
         * @param[in] camera pylon object representing the camera
         * @param[in] grabResult pylon object holding the image
         * @overload overloaded from pylon sdk
         */
        void OnImageGrabbed(Pylon::CInstantCamera &camera, const Pylon::CGrabResultPtr &grabResult) override;
        /**
         * @brief returns 4k image on demand
         * @return cv::Mat 4k image for detection
         * @warning It is thread safe
         */
        cv::Mat get4kColorImage() const;

    private:
        /**
         * @brief hd and color images objects
         */
        cv::Mat _color_4k_image, _color_hd_image;
        /**
         * @brief mutex to protect service shared resource during call
         */
        mutable std::mutex _4k_color_mutex;
        /**
         * @brief security color image publisher
         */
        rclcpp::Publisher<Image>::SharedPtr _bgr_color_publisher;
    };
    /**
     * @class LeftMonoImageEventHandler to grab moono image on callback
     * @ingroup basler
     * @brief called to grab mono, rotate 4k image
     */
    class LeftMonoImageEventHandler : public Pylon::CImageEventHandler
    {
    public:
        /**
         * @brief Construct a new Left Mono Image Event Handler object
         */
        LeftMonoImageEventHandler() = default;
        /**
         * @brief Construct a new Left Mono Image Event Handler object
         * @relatesalso RightMonoEventHandler
         */
        LeftMonoImageEventHandler(const LeftMonoImageEventHandler &) = delete;
        /**
         * @brief copys the image handler object
         * @return LeftMonoImageEventHandler&
         */
        LeftMonoImageEventHandler &operator=(const LeftMonoImageEventHandler &) = delete;
        /**
         * @brief Destroy the Left Mono Image Event Handler object
         */
        virtual ~LeftMonoImageEventHandler() = default;
        /**
         * @brief image grab callback is used to read mono image
         * @param[in] camera pylon object representing the camera
         * @param[in] grabResult pylon object holding the image
         * @overload overloaded from pylon sdk
         */
        void OnImageGrabbed(Pylon::CInstantCamera &camera, const Pylon::CGrabResultPtr &grabResult) override;
        /**
         * @brief Get the Left Mono Image object
         * @return cv::Mat 4k left mono image
         */
        cv::Mat getLeftMonoImage() const;

    private:
        /**
         * @brief 4k mono image object
         */
        cv::Mat _left_mono_image;
        /**
         * @brief mutex to protect service shared resource during call
         */
        mutable std::mutex _left_mono_mutex;
    };
    /**
     * @class RightMonoEventHandler to grab moono image on callback
     * @ingroup basler
     * @brief called to grab mono, rotate 4k image
     */
    class RightMonoEventHandler : public Pylon::CImageEventHandler
    {
    public:
        /**
         * @brief Construct a new Right Mono Event Handler object
         */
        RightMonoEventHandler() = default;
        /**
         * @brief Construct a new Right Mono Event Handler object
         */
        RightMonoEventHandler(const RightMonoEventHandler &) = delete;
        /**
         * @brief copys event handler object
         * @return RightMonoEventHandler&
         */
        RightMonoEventHandler &operator=(const RightMonoEventHandler &) = delete;
        /**
         * @brief Destroy the Right Mono Event Handler object
         */
        virtual ~RightMonoEventHandler() = default;
        /**
         * @brief image grab callback is used to read mono image
         * @param[in] camera pylon object representing the camera
         * @param[in] grabResult pylon object holding the image
         * @overload overloaded from pylon sdk
         */
        void OnImageGrabbed(Pylon::CInstantCamera &camera, const Pylon::CGrabResultPtr &grabResult) override;
         /**
         * @brief Get the right Mono Image object
         * @return cv::Mat 4k right mono image
         */
        cv::Mat getRightMonoimage() const;

    private:
        /**
         * @brief 4k mono image object
         */
        cv::Mat _right_mono_image;
        /**
         * @brief mutex to protect service shared resource during call
         */
        mutable std::mutex _right_mono_mutex;
    };
} // namespace basler
/** @} */ // end of group
#endif // __BASLER_IMAGE_HANDLER_HPP__
