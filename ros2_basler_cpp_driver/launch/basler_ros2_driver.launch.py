from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    container = ComposableNodeContainer(
        name='basler_ros2_driver_container',
        namespace='basler',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='basler_ros2_driver',
                plugin='basler::BaslerROS2Driver'
                # remappings=[
                #     # ('/group_1/mono_1', '/left/image_raw'),
                #     # ('/group_1/mono_1/camera_info', '/left/camera_info'),
                #     # ('/group_1/mono_2', '/right/image_raw'),
                #     # ('/group_1/mono_2/camera_info', '/right/camera_info'),
                #     # ('/group_1/color','/left/image_raw_color')
                #     ]
            ),
        ],
        output='screen',
        # prefix=['xterm -e gdb -ex run --args'],

    )
    return LaunchDescription([container])
