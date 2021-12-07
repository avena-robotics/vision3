// _PYLON_
#include "basler_ros2_driver/basler_image_handlers.hpp"

namespace basler
{
    cv::Mat ColorImageEventHandler::get4kColorImage() const
    {
        std::shared_lock lock(_4k_color_mutex);
        return _color_4k_image;
    }
    cv::Mat ColorImageEventHandler::getHDColorImage() const
    {
        std::shared_lock lock(_hd_color_mutex);
        return _color_hd_image;
    }
    void ColorImageEventHandler::OnImageGrabbed(Pylon::CInstantCamera & /*camera*/, const Pylon::CGrabResultPtr &grabResult)
    {
        if (grabResult->GrabSucceeded())
        {
            auto color_4k_image = cv::Mat(grabResult->GetHeight(), grabResult->GetWidth(), CV_8UC3, reinterpret_cast<uint8_t *>(grabResult->GetBuffer()));
            {
                std::unique_lock lock(_4k_color_mutex);
                cv::rotate(color_4k_image, _color_4k_image, cv::ROTATE_90_COUNTERCLOCKWISE);
            }
            {
                std::unique_lock hd_lock(_hd_color_mutex);
                std::shared_lock lock(_4k_color_mutex);
                cv::resize(_color_4k_image, _color_hd_image, cv::Size(static_cast<uint16_t>(_color_4k_image.size().width / 3), static_cast<uint16_t>(_color_4k_image.size().height / 3)), cv::INTER_AREA);
            }
        }
    }
    cv::Mat LeftMonoImageEventHandler::getLeftMonoImage() const
    {
        std::shared_lock lock(_left_mono_mutex);
        return _left_mono_image;
    }
    void LeftMonoImageEventHandler::OnImageGrabbed(Pylon::CInstantCamera & /*camera*/, const Pylon::CGrabResultPtr &grabResult)
    {
        if (grabResult->GrabSucceeded())
        {
            auto left_mono_image = cv::Mat(grabResult->GetHeight(), grabResult->GetWidth(), CV_8UC1, reinterpret_cast<uint8_t *>(grabResult->GetBuffer()));
            {
                std::unique_lock lock(_left_mono_mutex);
                cv::rotate(left_mono_image, _left_mono_image, cv::ROTATE_90_COUNTERCLOCKWISE);
            }
        }
    }
    cv::Mat RightMonoEventHandler::getRightMonoimage() const
    {
        std::shared_lock lock(_right_mono_mutex);
        return _right_mono_image;
    }
    void RightMonoEventHandler::OnImageGrabbed(Pylon::CInstantCamera & /*camera*/, const Pylon::CGrabResultPtr &grabResult)
    {
        if (grabResult->GrabSucceeded())
        {
            auto right_mono_image = cv::Mat(grabResult->GetHeight(), grabResult->GetWidth(), CV_8UC1, reinterpret_cast<uint8_t *>(grabResult->GetBuffer()));
            {
                std::unique_lock lock(_right_mono_mutex);
                cv::rotate(right_mono_image, _right_mono_image, cv::ROTATE_90_COUNTERCLOCKWISE);
            }
        }
    }
}