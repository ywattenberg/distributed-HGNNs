cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(torchtest)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(MACOSX TRUE)
endif()

message(STATUS "${CMAKE_CURRENT_LIST_DIR}")

set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_LIST_DIR}/libs/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Use homebrew when on macos
if (MACOSX)
    set(MAKE_PREFIX_PATH /opt/homebrew/opt/openblas/lib)
endif()

find_package(BLAS REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BLAS_CXX_FLAGS}")

# add_executable(example example.cpp)
# target_link_libraries(example "${TORCH_LIBRARIES}" "${BLAS_LIBRARIES}")
# set_property(TARGET example PROPERTY CXX_STANDARD 17)

add_executable(${PROJECT_NAME} main.cpp model/model.h model/model.cpp trainer/trainer.h trainer/trainer.cpp)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}" "${BLAS_LIBRARIES}")
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)