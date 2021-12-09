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

    class ColorImageEventHandler : public Pylon::CImageEventHandler
    {
    public:
        explicit ColorImageEventHandler(const rclcpp::Publisher<Image>::SharedPtr &hd_image_publihser);
        ColorImageEventHandler(const ColorImageEventHandler &) = delete;
        ColorImageEventHandler &operator=(const ColorImageEventHandler &) = delete;
        virtual ~ColorImageEventHandler() = default;
        void OnImageGrabbed(Pylon::CInstantCamera &camera, const Pylon::CGrabResultPtr &grabResult) override;
        cv::Mat get4kColorImage() const;
        // cv::Mat getHDColorImage() const;

    private:
        cv::Mat _color_4k_image, _color_hd_image;
        // mutable std::mutex _hd_color_mutex, _4k_color_mutex;
        mutable std::mutex _4k_color_mutex;
        // mutable std::shared_mutex _hd_color_mutex,_4k_color_mutex;
        // _ROS_
        // publishers
        rclcpp::Publisher<Image>::SharedPtr _bgr_color_publisher;
        std::chrono::time_point<std::chrono::system_clock> _color_prev_time;
    };
    class LeftMonoImageEventHandler : public Pylon::CImageEventHandler
    {
    public:
        LeftMonoImageEventHandler();
        LeftMonoImageEventHandler(const LeftMonoImageEventHandler &) = delete;
        LeftMonoImageEventHandler &operator=(const LeftMonoImageEventHandler &) = delete;
        virtual ~LeftMonoImageEventHandler() = default;
        void OnImageGrabbed(Pylon::CInstantCamera &camera, const Pylon::CGrabResultPtr &grabResult) override;
        cv::Mat getLeftMonoImage() const;

    private:
        cv::Mat _left_mono_image;
        mutable std::mutex _left_mono_mutex;
        std::chrono::time_point<std::chrono::system_clock> _left_mono_prev_time;
        // mutable std::shared_mutex _left_mono_mutex;
    };
    class RightMonoEventHandler : public Pylon::CImageEventHandler
    {
    public:
        RightMonoEventHandler();
        RightMonoEventHandler(const RightMonoEventHandler &) = delete;
        RightMonoEventHandler &operator=(const RightMonoEventHandler &) = delete;
        virtual ~RightMonoEventHandler() = default;
        void OnImageGrabbed(Pylon::CInstantCamera &camera, const Pylon::CGrabResultPtr &grabResult) override;
        cv::Mat getRightMonoimage() const;

    private:
        cv::Mat _right_mono_image;
        mutable std::mutex _right_mono_mutex;
        // mutable std::shared_mutex _right_mono_mutex;
        std::chrono::time_point<std::chrono::system_clock> _right_mono_prev_time;

    };
} // namespace basler

#endif // __BASLER_IMAGE_HANDLER_HPP__
