// _PYLON_
#include "basler_ros2_driver/basler_image_handlers.hpp"

namespace basler
{
    ColorImageEventHandler::ColorImageEventHandler(const rclcpp::Publisher<Image>::SharedPtr &hd_image_publihser) : _bgr_color_publisher(hd_image_publihser)
    {
    }
    cv::Mat ColorImageEventHandler::get4kColorImage() const
    {
        std::lock_guard lock(_4k_color_mutex);
        cv::Mat temp_4k_color_image = _color_4k_image.clone();
        return temp_4k_color_image;
    }
    void ColorImageEventHandler::OnImageGrabbed(Pylon::CInstantCamera & /*camera*/, const Pylon::CGrabResultPtr &grabResult)
    {
        try
        {
            if (grabResult->GrabSucceeded())
            {
                auto color_4k_image = cv::Mat(grabResult->GetHeight(), grabResult->GetWidth(), CV_8UC3, reinterpret_cast<uint8_t *>(grabResult->GetBuffer()));
                {
                    std::lock_guard lock(_4k_color_mutex);
                    if (!color_4k_image.empty())
                    {
                        cv::rotate(color_4k_image, _color_4k_image, cv::ROTATE_90_COUNTERCLOCKWISE);
                    }
                }
                if (!_color_4k_image.empty())
                {
                    cv::resize(_color_4k_image, _color_hd_image, cv::Size(static_cast<uint16_t>(_color_4k_image.size().width / 3), static_cast<uint16_t>(_color_4k_image.size().height / 3)), cv::INTER_AREA);
                    if (_color_hd_image.size().width > 0)
                    {
                        auto hd_color_image_msg = std::make_shared<sensor_msgs::msg::Image>();
                        hd_color_image_msg->header.frame_id = "/basler/color/image_raw";
                        cv_bridge::CvImagePtr cv_ptr = std::make_shared<cv_bridge::CvImage>(hd_color_image_msg->header, sensor_msgs::image_encodings::BGR8, _color_hd_image);
                        try
                        {
                            hd_color_image_msg = cv_ptr->toImageMsg();
                        }
                        catch (cv_bridge::Exception &e)
                        {
                            std::cerr << e.what() << std::endl;
                            return;
                        }
                        if (_bgr_color_publisher->get_subscription_count() > 0 && hd_color_image_msg->width > 0)
                        {
                            _bgr_color_publisher->publish(*hd_color_image_msg);
                        }
                    }
                }
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
        }
    }
    cv::Mat LeftMonoImageEventHandler::getLeftMonoImage() const
    {
        std::lock_guard lock(_left_mono_mutex);
        cv::Mat temp_left_mono_img = _left_mono_image.clone();
        return temp_left_mono_img;
    }
    void LeftMonoImageEventHandler::OnImageGrabbed(Pylon::CInstantCamera & /*camera*/, const Pylon::CGrabResultPtr &grabResult)
    {
        try
        {
            if (grabResult->GrabSucceeded())
            {
                auto left_mono_image = cv::Mat(grabResult->GetHeight(), grabResult->GetWidth(), CV_8UC1, reinterpret_cast<uint8_t *>(grabResult->GetBuffer()));
                {
                    std::lock_guard lock(_left_mono_mutex);
                    if (!left_mono_image.empty())
                    {
                        cv::rotate(left_mono_image, _left_mono_image, cv::ROTATE_90_COUNTERCLOCKWISE);
                    }
                }
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
        }
    }

    cv::Mat RightMonoEventHandler::getRightMonoimage() const
    {
        std::lock_guard lock(_right_mono_mutex);
        cv::Mat temp_right_mono_image = _right_mono_image.clone();
        return temp_right_mono_image;
    }
    void RightMonoEventHandler::OnImageGrabbed(Pylon::CInstantCamera & /*camera*/, const Pylon::CGrabResultPtr &grabResult)
    {
        try
        {
            if (grabResult->GrabSucceeded())
            {
                auto right_mono_image = cv::Mat(grabResult->GetHeight(), grabResult->GetWidth(), CV_8UC1, reinterpret_cast<uint8_t *>(grabResult->GetBuffer()));
                {
                    std::lock_guard lock(_right_mono_mutex);
                    if (!right_mono_image.empty())
                    {
                        cv::rotate(right_mono_image, _right_mono_image, cv::ROTATE_90_COUNTERCLOCKWISE);
                    }
                }
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
        }
    }
}