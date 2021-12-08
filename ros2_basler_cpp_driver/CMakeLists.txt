cmake_minimum_required(VERSION 3.5)
project(basler_ros2_driver)


# Default to C++17
set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_BUILD_TYPE RELEASE)
set(CMAKE_BUILD_TYPE DEBUG)
add_compile_options(-Wall -Wextra -Wpedantic)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(custom_interfaces REQUIRED)
find_package(std_srvs REQUIRED)
find_package(sensor_msgs REQUIRED)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindPylon.cmake") # find_package(Pylon REQUIRED)
find_package(OpenCV REQUIRED )
find_package(ament_index_cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED )

add_library(${PROJECT_NAME} SHARED src/basler_driver.cpp src/basler_image_handlers.cpp)
target_link_libraries(
    ${PROJECT_NAME}
    ${Pylon_LIBRARIES}
    ${OpenCV_LIBS}
)
target_include_directories(${PROJECT_NAME} PUBLIC
${Pylon_INCLUDE_DIRS}
${OpenCV_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(
  ${PROJECT_NAME}
  "rclcpp"
  "rclcpp_components"
  "custom_interfaces"
  "std_srvs"
  "sensor_msgs"
  "image_transport"
  "cv_bridge"
)
rclcpp_components_register_node(${PROJECT_NAME} PLUGIN "basler::BaslerROS2Driver" EXECUTABLE basler_ros2_driver_main)

install(TARGETS
${PROJECT_NAME}
ARCHIVE DESTINATION lib
LIBRARY DESTINATION lib
RUNTIME DESTINATION bin)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
