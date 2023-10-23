cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(torchtest)

message(STATUS "${CMAKE_CURRENT_LIST_DIR}")

set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_LIST_DIR}/libs/libtorch")
find_package(Torch REQUIRED)
find_package(BLAS REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(example example.cpp)
target_link_libraries(example "${TORCH_LIBRARIES}" "${BLAS_LIBRARIES}")
set_property(TARGET example PROPERTY CXX_STANDARD 17)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}" "${BLAS_LIBRARIES}")
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)