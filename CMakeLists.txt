cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# Initialize for C++ and CUDA languages
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(kinfu_imp LANGUAGES CXX CUDA)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# For CUDA
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message("Cuda version: " ${CUDA_VERSION})
    CUDA_ADD_EXECUTABLE(kinfu_imp
    src/system_interface.cpp
    src/system_interface.h
    src/data_initialization.cpp
    src/projective_tsdf.cu
    src/voxel_traversal.cu
    src/raw_vertices.cu
    src/sensor_pose_estimation.cu
    src/cuda_functions.cuh
    )
endif()

target_link_libraries(kinfu_imp ${OpenCV_LIBS} )
