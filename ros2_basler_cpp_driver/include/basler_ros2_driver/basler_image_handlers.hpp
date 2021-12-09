#ifndef __BASLER_IMAGE_HANDLER_HPP__
#define __BASLER_IMAGE_HANDLER_HPP__
// __HEADERs__

// _PYLON_
#include <pylon/PylonIncludes.h>

// _OPENCV_
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>

// _CPP_
#include <shared_mutex>

// _ROS_
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>

namespace basler
{
    // __Deductions__

    // _ROS_
    using Image = sensor_msgs::msg::Image;
    /**
     * @brief
     *
     */
    class ColorImageEventHandler : public Pylon::CImageEventHandler
    {
    public:
        /**
         * @brief Construct a new Color Image Event Handler object
         *
         * @param hd_image_publihser
         */
        explicit ColorImageEventHandler(const rclcpp::Publisher<Image>::SharedPtr &hd_image_publihser);
        /**
         * @brief Construct a new Color Image Event Handler object
         *
         */
        ColorImageEventHandler(const ColorImageEventHandler &) = delete;
        /**
         * @brief
         *
         * @return ColorImageEventHandler&
         */
        ColorImageEventHandler &operator=(const ColorImageEventHandler &) = delete;
        /**
         * @brief Destroy the Color Image Event Handler object
         *
         */
        virtual ~ColorImageEventHandler() = default;
        /**
         * @brief
         *
         * @param camera
         * @param grabResult
         */
        void OnImageGrabbed(Pylon::CInstantCamera &camera, const Pylon::CGrabResultPtr &grabResult) override;
        /**
         * @brief
         *
         * @return cv::Mat
         */
        cv::Mat get4kColorImage() const;

    private:
        cv::Mat _color_4k_image, _color_hd_image;
        mutable std::mutex _4k_color_mutex;
        // _ROS_
        // publishers
        rclcpp::Publisher<Image>::SharedPtr _bgr_color_publisher;
    };
    class LeftMonoImageEventHandler : public Pylon::CImageEventHandler
    {
    public:
        /**
         * @brief Construct a new Left Mono Image Event Handler object
         *
         */
        LeftMonoImageEventHandler() = default;
        /**
         * @brief Construct a new Left Mono Image Event Handler object
         *
         */
        LeftMonoImageEventHandler(const LeftMonoImageEventHandler &) = delete;
        /**
         * @brief
         *
         * @return LeftMonoImageEventHandler&
         */
        LeftMonoImageEventHandler &operator=(const LeftMonoImageEventHandler &) = delete;
        /**
         * @brief Destroy the Left Mono Image Event Handler object
         *
         */
        virtual ~LeftMonoImageEventHandler() = default;
        /**
         * @brief
         *
         * @param camera
         * @param grabResult
         */
        void OnImageGrabbed(Pylon::CInstantCamera &camera, const Pylon::CGrabResultPtr &grabResult) override;
        /**
         * @brief Get the Left Mono Image object
         *
         * @return cv::Mat
         */
        cv::Mat getLeftMonoImage() const;

    private:
        cv::Mat _left_mono_image;
        mutable std::mutex _left_mono_mutex;
    };
    /**
     * @brief
     *
     */
    class RightMonoEventHandler : public Pylon::CImageEventHandler
    {
    public:
        /**
         * @brief Construct a new Right Mono Event Handler object
         *
         */
        RightMonoEventHandler() = default;
        /**
         * @brief Construct a new Right Mono Event Handler object
         *
         */
        RightMonoEventHandler(const RightMonoEventHandler &) = delete;
        /**
         * @brief
         *
         * @return RightMonoEventHandler&
         */
        RightMonoEventHandler &operator=(const RightMonoEventHandler &) = delete;
        /**
         * @brief Destroy the Right Mono Event Handler object
         *
         */
        virtual ~RightMonoEventHandler() = default;
        /**
         * @brief
         *
         * @param camera
         * @param grabResult
         */
        void OnImageGrabbed(Pylon::CInstantCamera &camera, const Pylon::CGrabResultPtr &grabResult) override;
        /**
         * @brief Get the Right Monoimage object
         * 
         * @return cv::Mat 
         */
        cv::Mat getRightMonoimage() const;

    private:
        cv::Mat _right_mono_image;
        mutable std::mutex _right_mono_mutex;
    };
} // namespace basler

#endif // __BASLER_IMAGE_HANDLER_HPP__
