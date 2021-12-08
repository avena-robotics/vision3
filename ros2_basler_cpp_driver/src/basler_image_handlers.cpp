// _PYLON_
#include "basler_ros2_driver/basler_image_handlers.hpp"

namespace basler
{
    cv::Mat ColorImageEventHandler::get4kColorImage() const
    {
        // std::shared_lock lock(_4k_color_mutex);
        std::lock_guard lock(_4k_color_mutex);
        cv::Mat temp_4k_color_image = _color_4k_image.clone();
        return temp_4k_color_image;
    }
    cv::Mat ColorImageEventHandler::getHDColorImage() const
    {
        // std::shared_lock lock(_hd_color_mutex);
        std::lock_guard lock(_hd_color_mutex);
        cv::Mat temp_color_hd_image = _color_hd_image.clone();
        return temp_color_hd_image;
    }
    void ColorImageEventHandler::OnImageGrabbed(Pylon::CInstantCamera & /*camera*/, const Pylon::CGrabResultPtr &grabResult)
    {
        try
        {
            if (grabResult->GrabSucceeded())
            {
                auto color_4k_image = cv::Mat(grabResult->GetHeight(), grabResult->GetWidth(), CV_8UC3, reinterpret_cast<uint8_t *>(grabResult->GetBuffer()));
                {
                    // std::unique_lock lock(_4k_color_mutex);
                    std::lock_guard lock(_4k_color_mutex);
                    if (!color_4k_image.empty())
                    {
                        cv::rotate(color_4k_image, _color_4k_image, cv::ROTATE_90_COUNTERCLOCKWISE);
                    }
                }
            }
            {
                // std::unique_lock hd_lock(_hd_color_mutex);
                // std::shared_lock lock(_4k_color_mutex);
                std::lock_guard lock(_hd_color_mutex);
                if (!_color_4k_image.empty())
                {
                    cv::resize(_color_4k_image, _color_hd_image, cv::Size(static_cast<uint16_t>(_color_4k_image.size().width / 3), static_cast<uint16_t>(_color_4k_image.size().height / 3)), cv::INTER_AREA);
                }
            }
            // else
            // {
            //     std::cerr << "Grab Failed" << '\n';
            //     return;
            // }
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
        }
    }
    cv::Mat LeftMonoImageEventHandler::getLeftMonoImage() const
    {
        // std::shared_lock lock(_left_mono_mutex);
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
                    // std::unique_lock lock(_left_mono_mutex);
                    std::lock_guard lock(_left_mono_mutex);
                    if (!left_mono_image.empty())
                    {
                        cv::rotate(left_mono_image, _left_mono_image, cv::ROTATE_90_COUNTERCLOCKWISE);
                    }
                }
            }
            // else
            // {
            //     std::cerr << "Grab Failed" << '\n';
            //     return;
            // }
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
        }
    }

    cv::Mat RightMonoEventHandler::getRightMonoimage() const
    {
        // std::shared_lock lock(_right_mono_mutex);
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
                    // std::unique_lock lock(_right_mono_mutex);
                    std::lock_guard lock(_right_mono_mutex);
                    if (!right_mono_image.empty())
                    {
                        cv::rotate(right_mono_image, _right_mono_image, cv::ROTATE_90_COUNTERCLOCKWISE);
                    }
                }
            }
            // else
            // {
            //     std::cerr << "Grab Failed" << '\n';
            //     return;
            // }
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
        }
    }
}