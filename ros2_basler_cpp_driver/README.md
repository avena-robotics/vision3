# ROS2 Basler CPP Driver

## Source ROS 
1. source install/setup.bash 

2. source /opt/ros/foxy/setup.bash

# Launch Camera controller Server

3. ros2 launch basler_ros2_driver basler_ros2_driver.launch.py 

# Call Camera controller Services

4. ros2 service call /open_basler_cameras std_srvs/srv/Trigger {}

5. ros2 service call /get_basler_all_images custom_interfaces/srv/GetAllImages {}

6. ros2 service call /get_basler_mono_images custom_interfaces/srv/GetMonoImages {}

7. ros2 service call /get_basler_color_image custom_interfaces/srv/GetColorImage {}

8. ros2 service call /close_basler_cameras std_srvs/srv/Trigger {}

# (Optional) View hd Color Image

9. rviz2 -d ros2_basler_cpp_driver/config/basler.rviz

# (Optional) check fps of color stream

10. ros2 topic hz /basler/color/image_raw 
