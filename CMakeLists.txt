cmake_minimum_required(VERSION 2.8.3 FATAL_ERROR)
project(cylinders_localization)

## Compile as C++11, supported in ROS Kinetic and newer
##add_compile_options(-std=c++11)

set(CMAKE_BUILD_TYPE "Debug")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  cv_bridge
  image_transport
  sensor_msgs
  message_filters
  tf
)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED) # Ceres: version 1.12, so Eigen version can < 3.1 without being changed.

include_directories(
  include
	${catkin_INCLUDE_DIRS} 
	${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS})

#generate_messages()
catkin_package(
   CATKIN_DEPENDS roscpp
   DEPENDS EIGEN3 PCL
   INCLUDE_DIRS include
)

add_executable( cylinders_localization src/cylinders_localization.cpp)
target_link_libraries(cylinders_localization ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${CERES_LIBRARIES})
