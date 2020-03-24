cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(postproc)

set(CMAKE_PREFIX_PATH "/home/ppotrykus/bus_detector/cpp/libtorch" CACHE PATH "libtorch library directory")
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/home/ppotrykus/anaconda3/envs/libtorch/" )
# set(CUDA_TOOLKIT_ROOT_DIR "$ENV{CUDA_TOOLKIT_ROOT_DIR}" CACHE PATH "CUDA install path")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-10.1" CACHE PATH "CUDA install path")

find_package(Torch REQUIRED)
find_package(OpenCV 3.4.2 EXACT REQUIRED)

# include_directories(${OpenCV_INCLUDE_DIRS})
# include_directories(${TORCH_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(postproc main.cpp)
target_link_libraries(postproc "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
set_property(TARGET postproc PROPERTY CXX_STANDARD 14)

