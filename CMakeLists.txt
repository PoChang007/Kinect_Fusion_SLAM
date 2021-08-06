cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# Initialize for C++ and CUDA languages
set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(kinfu_imp LANGUAGES CXX CUDA)

find_package(OpenCV 4.1 REQUIRED)
find_package(PCL 1.7 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})

link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${PCL_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})
add_definitions(${OpenCV_DEFINITIONS})

list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

# fix potential warning causing from error in VTK compile definitions
get_property(COMPILE_DEFINITIONS_PROPERTY DIRECTORY PROPERTY COMPILE_DEFINITIONS)
list(FIND COMPILE_DEFINITIONS_PROPERTY "vtkRenderingCore_AUTOINIT=3(vtkInteractionStyle,vtkRenderingFreeType,vtkRenderingOpenGL2)" INDEX)
if(NOT INDEX EQUAL -1)
    list(REMOVE_AT COMPILE_DEFINITIONS_PROPERTY ${INDEX})
    list(APPEND COMPILE_DEFINITIONS_PROPERTY "vtkRenderingCore_AUTOINIT=3(vtkInteractionStyle ,vtkRenderingFreeType ,vtkRenderingOpenGL2 )")
    #replace with the same properties with spaces after the macro name
    set_property(DIRECTORY PROPERTY COMPILE_DEFINITIONS ${COMPILE_DEFINITIONS_PROPERTY})
endif()

# For CUDA
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message("Cuda version: " ${CUDA_VERSION})
    CUDA_ADD_EXECUTABLE(kinfu_imp
    src/system_interface.cpp
    src/kinfu_pipeline.cpp
    src/kinfu_pipeline.h
    src/system_utility.cpp
    src/system_utility.h
    src/threeD_viewer.cpp
    src/threeD_viewer.h
    src/parallel_processing/parallel_processing.h
    src/parallel_processing/projective_tsdf.cu
    src/parallel_processing/voxel_traversal.cu
    src/parallel_processing/raw_vertices.cu
    src/parallel_processing/sensor_pose_estimation.cu
    src/parallel_processing/bilateral_filtering.cu
    src/parallel_processing/cuda_functions.cuh
    )
endif()

target_link_libraries (kinfu_imp ${OpenCV_LIBRARIES})
target_link_libraries (kinfu_imp ${PCL_LIBRARIES})