import json
import time
import pyrealsense2 as rs


def open_intel_camera(config_filename: str):
    """Opens Intel Realsense camera.

    This function is responsible for opening Intel Realsense camera 
    with provided path to configuration file which consists serial number,
    resolutions for depth and color, framerates and laser power.

    Args:
        config_filename: path to configuration file
    
    Returns:
        Tuple with 2 elements:
            - handle to opened camera,
            - dictionary with intrinsic parameters of color sensor (i.e. fx, fy, cx, cy, width, height)
    """
    camera_settings = json.load(open(config_filename))
    camera_serial = camera_settings["serial_number"]
    camera_config = rs.config()
    camera_config.enable_stream(rs.stream.depth, int(camera_settings["stream-depth-width"]),
                                int(camera_settings["stream-depth-height"]), rs.format.z16,
                                int(camera_settings["stream-depth-fps"]))
    camera_config.enable_stream(rs.stream.color, int(camera_settings["stream-color-width"]),
                                int(camera_settings["stream-color-height"]), rs.format.rgb8,
                                int(camera_settings["stream-color-fps"]))
    camera_config.enable_device(camera_serial)
    camera_handle = rs.pipeline()
    camera_pipeline_profile = camera_handle.start(camera_config)
    camera_device = camera_pipeline_profile.get_device()

    with open("/home/avena/software/librealsense/wrappers/python/examples/box_dimensioner_multicam/HighResHighAccuracyPreset.json", 'r') as file:
    # with open("/home/avena/vision3/high_accuracy_preset.json", 'r') as file:
        json_text = file.read().strip()
    advanced_mode = rs.rs400_advanced_mode(camera_device)
    advanced_mode.load_json(json_text)

    sensors = camera_device.query_sensors()
    for sensor in sensors:
        if rs.sensor.as_depth_sensor(sensor):
            sensor.set_option(rs.option.emitter_enabled, 1)
            sensor.set_option(rs.option.laser_power, int(camera_settings["controls-laserpower"]))
            sensor.set_option(rs.option.enable_auto_exposure, 1)

        elif rs.sensor.as_color_sensor(sensor):
            sensor.set_option(rs.option.enable_auto_exposure, 1)
            sensor.set_option(rs.option.enable_auto_white_balance, 1)

    # Read intrinsic
    camera_pipeline_profile = camera_handle.get_active_profile()
    color_profile = camera_pipeline_profile.get_stream(rs.stream.color)
    color_intrinsic = color_profile.as_video_stream_profile().get_intrinsics()

    print('Waiting for a few seconds to start camera...')
    time.sleep(1.5)

    return camera_handle, {"fx": color_intrinsic.fx, "fy": color_intrinsic.fy, 
                           "cx": color_intrinsic.ppx, "cy": color_intrinsic.ppy, 
                           "height": color_intrinsic.height, "width": color_intrinsic.width}
    