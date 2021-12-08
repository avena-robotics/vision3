#ifndef __BASLER_IMAGE_HANDLER_HPP__
#define __BASLER_IMAGE_HANDLER_HPP__
#include <pylon/PylonIncludes.h>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <shared_mutex>
namespace basler
{

    class ColorImageEventHandler : public Pylon::CImageEventHandler
    {
    public:
        ColorImageEventHandler() = default;
        ColorImageEventHandler(const ColorImageEventHandler &) = delete;
        ColorImageEventHandler &operator=(const ColorImageEventHandler &) = delete;
        virtual ~ColorImageEventHandler() = default;
        void OnImageGrabbed(Pylon::CInstantCamera &camera, const Pylon::CGrabResultPtr &grabResult) override;
        cv::Mat get4kColorImage() const;
        cv::Mat getHDColorImage() const;

    private:
        cv::Mat _color_4k_image, _color_hd_image;
        mutable std::mutex _hd_color_mutex, _4k_color_mutex;
        // mutable std::shared_mutex _hd_color_mutex,_4k_color_mutex;
    };
    class LeftMonoImageEventHandler : public Pylon::CImageEventHandler
    {
    public:
        LeftMonoImageEventHandler() = default;
        LeftMonoImageEventHandler(const LeftMonoImageEventHandler &) = delete;
        LeftMonoImageEventHandler &operator=(const LeftMonoImageEventHandler &) = delete;
        virtual ~LeftMonoImageEventHandler() = default;
        void OnImageGrabbed(Pylon::CInstantCamera &camera, const Pylon::CGrabResultPtr &grabResult) override;
        cv::Mat getLeftMonoImage() const;

    private:
        cv::Mat _left_mono_image;
        mutable std::mutex _left_mono_mutex;
        // mutable std::shared_mutex _left_mono_mutex;
    };
    class RightMonoEventHandler : public Pylon::CImageEventHandler
    {
    public:
        RightMonoEventHandler() = default;
        RightMonoEventHandler(const RightMonoEventHandler &) = delete;
        RightMonoEventHandler &operator=(const RightMonoEventHandler &) = delete;
        virtual ~RightMonoEventHandler() = default;
        void OnImageGrabbed(Pylon::CInstantCamera &camera, const Pylon::CGrabResultPtr &grabResult) override;
        cv::Mat getRightMonoimage() const;

    private:
        cv::Mat _right_mono_image;
        mutable std::mutex _right_mono_mutex;
        // mutable std::shared_mutex _right_mono_mutex;
    };
} // namespace basler

#endif // __BASLER_IMAGE_HANDLER_HPP__
