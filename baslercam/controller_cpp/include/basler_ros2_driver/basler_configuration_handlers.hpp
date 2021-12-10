/**
 * This file is part of basler_ros2_driver ROS2 Package
 * License declaration
 * @file  basler_configuration_handlers
 * @name basler_configuration_handlers.hpp
 * @addtogroup basler_ros2_driver
 * @author Hasan Farag
 * @link [3d-processing] https://app.developerhub.io/robotics/v3.0/robotics/3d-processing
 * @ref See also pylon docs for daa3840-45um and daa3840-45uc
 */
#ifndef __BASLER_CONFIGUARTION_HANDLER_HPP__
#define __BASLER_CONFIGUARTION_HANDLER_HPP__

/**
 *  @defgroup Headers file headers
 */
/** @defgroup Pylon header files for basler sdk
 * @ingroup headers
 *  @{
 */
#include <pylon/ConfigurationEventHandler.h>
#include <pylon/ParameterIncludes.h>
/** @} */ // end of Pylon
    /**
     * @class MonoCameraConfigurationHandler  Mono Camera configurator
     * @brief  represents Mono cameras Configuration Event Handler
     */
class MonoCameraConfigurationHandler : public Pylon::CConfigurationEventHandler
{
public:
    /**
     * @brief configure mono cameras on open camera event
     * @param[in] camera  pylon device
     */
    void OnOpened(Pylon::CInstantCamera &camera)
    {
        try
        {
            const uint8_t basler_binning = 1;
            const float basler_mono_brightness = 0.3;

            auto &cam_nodemap = camera.GetNodeMap();
            auto width = Pylon::CIntegerParameter(cam_nodemap, "Width").GetValue();
            auto height = Pylon::CIntegerParameter(cam_nodemap, "Height").GetValue();

            Pylon::CIntegerParameter(cam_nodemap, "DeviceLinkThroughputLimit").SetValue(320000000);
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
    /**
     * @class ColorCameraConfigurationHandler  color Camera configurator
     * @brief  represents color cameras Configuration Event Handler
     */
class ColorCameraConfigurationHandler : public Pylon::CConfigurationEventHandler
{
public:
   /**
     * @brief configure color cameras on open camera event
     * @param[in] camera  pylon device
     */
    void OnOpened(Pylon::CInstantCamera &camera)
    {
        try
        {
            const uint8_t basler_binning = 1;
            const float basler_color_brightness = 0.2;

            auto &cam_nodemap = camera.GetNodeMap();
            auto width = Pylon::CIntegerParameter(cam_nodemap, "Width").GetValue();
            auto height = Pylon::CIntegerParameter(cam_nodemap, "Height").GetValue();

            Pylon::CIntegerParameter(cam_nodemap, "DeviceLinkThroughputLimit").SetValue(419430400);
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

            if (Pylon::CEnumParameter(cam_nodemap, "PixelFormat").GetValue() != "BGR8")
                Pylon::CEnumParameter(cam_nodemap, "PixelFormat").SetValue("BGR8");
        }
        catch (const Pylon::GenericException &e)
        {
            throw RUNTIME_EXCEPTION("Could not apply configuration. const GenericException caught in OnOpened method msg=%hs", e.what());
        }
    }
};

#endif