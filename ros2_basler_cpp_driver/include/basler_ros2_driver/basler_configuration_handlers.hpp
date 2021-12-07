#ifndef __BASLER_CONFIGUARTION_HANDLER_HPP__
#define __BASLER_CONFIGUARTION_HANDLER_HPP__
#include <pylon/ConfigurationEventHandler.h>
#include <pylon/ParameterIncludes.h>

class MonoCameraConfigurationHandler : public Pylon::CConfigurationEventHandler
{
public:
    void OnOpened(Pylon::CInstantCamera &camera)
    {
        try
        {
            const uint8_t basler_binning = 1;
            const float basler_mono_brightness = 0.15;

            auto &cam_nodemap = camera.GetNodeMap();
            auto width = Pylon::CIntegerParameter(cam_nodemap, "Width").GetValue();
            auto height = Pylon::CIntegerParameter(cam_nodemap, "Height").GetValue();

            Pylon::CIntegerParameter(cam_nodemap, "DeviceLinkThroughputLimit").SetValue(320000000); // TODO: 45 fps
            Pylon::CEnumParameter(cam_nodemap, "AutoFunctionROISelector").SetValue("ROI1");
            Pylon::CIntegerParameter(cam_nodemap, "AutoFunctionROIOffsetX").SetValue(0);
            Pylon::CIntegerParameter(cam_nodemap, "AutoFunctionROIOffsetY").SetValue(0);
            Pylon::CIntegerParameter(cam_nodemap, "AutoFunctionROIWidth").SetValue(width / basler_binning);
            Pylon::CIntegerParameter(cam_nodemap, "AutoFunctionROIHeight").SetValue(height / basler_binning);
            Pylon::CBooleanParameter(cam_nodemap, "AutoFunctionROIUseBrightness").SetValue(true);

            if (Pylon::CEnumParameter(cam_nodemap, "GainAuto").GetValue() != "Continuous")
                Pylon::CEnumParameter(cam_nodemap, "GainAuto").SetValue("Continuous");

            if (Pylon::CEnumParameter(cam_nodemap, "ExposureAuto").GetValue() != "Continuous")
                Pylon::CEnumParameter(cam_nodemap, "ExposureAuto").SetValue("Continuous");

            if (Pylon::CEnumParameter(cam_nodemap, "BinningVerticalMode").GetValue() != "Average")
                Pylon::CEnumParameter(cam_nodemap, "BinningVerticalMode").SetValue("Average");

            if (Pylon::CEnumParameter(cam_nodemap, "BinningHorizontalMode").GetValue() != "Average")
                Pylon::CEnumParameter(cam_nodemap, "BinningHorizontalMode").SetValue("Average");

            if (GenApi_3_1_Basler_pylon::IsAvailable(Pylon::CIntegerParameter(cam_nodemap, "BinningHorizontal")))
            {
                if (Pylon::CIntegerParameter(cam_nodemap, "BinningHorizontal").GetValue() != basler_binning)
                    Pylon::CIntegerParameter(cam_nodemap, "BinningHorizontal").SetValue(basler_binning);
            }
            if (GenApi_3_1_Basler_pylon::IsAvailable(Pylon::CIntegerParameter(cam_nodemap, "BinningVertical")))
            {
                if (Pylon::CIntegerParameter(cam_nodemap, "BinningVertical").GetValue() != basler_binning)
                    Pylon::CIntegerParameter(cam_nodemap, "BinningVertical").SetValue(basler_binning);
            }
            if (Pylon::CFloatParameter(cam_nodemap, "AutoTargetBrightness").GetValue() != basler_mono_brightness)
                Pylon::CFloatParameter(cam_nodemap, "AutoTargetBrightness").SetValue(basler_mono_brightness);
        }
        catch (const Pylon::GenericException &e)
        {
            throw RUNTIME_EXCEPTION("Could not apply configuration. const GenericException caught in OnOpened method msg=%hs", e.what());
        }
    }
};
class ColorCameraConfigurationHandler : public Pylon::CConfigurationEventHandler
{
public:
    void OnOpened(Pylon::CInstantCamera &camera)
    {
        try
        {
            const uint8_t basler_binning = 1;
            const float basler_color_brightness = 0.2;

            auto &cam_nodemap = camera.GetNodeMap();
            auto width = Pylon::CIntegerParameter(cam_nodemap, "Width").GetValue();
            auto height = Pylon::CIntegerParameter(cam_nodemap, "Height").GetValue();

            Pylon::CIntegerParameter(cam_nodemap, "DeviceLinkThroughputLimit").SetValue(320000000); // TODO: 45 fps
            Pylon::CEnumParameter(cam_nodemap, "AutoFunctionROISelector").SetValue("ROI1");
            Pylon::CIntegerParameter(cam_nodemap, "AutoFunctionROIOffsetX").SetValue(0);
            Pylon::CIntegerParameter(cam_nodemap, "AutoFunctionROIOffsetY").SetValue(0);
            Pylon::CIntegerParameter(cam_nodemap, "AutoFunctionROIWidth").SetValue(width / basler_binning);
            Pylon::CIntegerParameter(cam_nodemap, "AutoFunctionROIHeight").SetValue(height / basler_binning);
            Pylon::CBooleanParameter(cam_nodemap, "AutoFunctionROIUseBrightness").SetValue(true);

            if (Pylon::CEnumParameter(cam_nodemap, "GainAuto").GetValue() != "Continuous")
                Pylon::CEnumParameter(cam_nodemap, "GainAuto").SetValue("Continuous");

            if (Pylon::CEnumParameter(cam_nodemap, "ExposureAuto").GetValue() != "Continuous")
                Pylon::CEnumParameter(cam_nodemap, "ExposureAuto").SetValue("Continuous");

            if (Pylon::CFloatParameter(cam_nodemap, "AutoTargetBrightness").GetValue() != basler_color_brightness)
                Pylon::CFloatParameter(cam_nodemap, "AutoTargetBrightness").SetValue(basler_color_brightness);

            Pylon::CEnumParameter(cam_nodemap, "PixelFormat").SetValue("BGR8");
        }
        catch (const Pylon::GenericException &e)
        {
            throw RUNTIME_EXCEPTION("Could not apply configuration. const GenericException caught in OnOpened method msg=%hs", e.what());
        }
    }
};

#endif