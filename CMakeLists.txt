cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(torchtest)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(MACOSX TRUE)
endif()

message(STATUS "${CMAKE_CURRENT_LIST_DIR}")

# find libtorch
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_LIST_DIR}/libs/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# find openblas, use homebrew when on macos
if (MACOSX)
    set(MAKE_PREFIX_PATH /opt/homebrew/opt/openblas/lib)
endif()

find_package(BLAS REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BLAS_CXX_FLAGS}")

# find the YAML-CPP package
find_package(yaml-cpp REQUIRED)
message(STATUS  "${YAML_CPP_INCLUDE_DIR}")

# find OpenMP (MacOS)
set(OpenMP_C "${CMAKE_C_COMPILER}")
set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/llvm/include")
set(OpenMP_C_LIB_NAMES "omp")
set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/llvm/include")
set(OpenMP_CXX_LIB_NAMES "omp")
set(OpenMP_omp_LIBRARY "/usr/local/Cellar/libomp/17.0.2/lib/libomp.dylib")

set(CMAKE_CXX_COMPILER "/usr/local/opt/llvm/bin/clang++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
include_directories("/usr/local/Cellar/libomp/17.0.2/include/omp.h")

find_package(OpenMP REQUIRED)

# # add example executable
# add_executable(example example.cpp)
# target_link_libraries(example "${TORCH_LIBRARIES}" "${BLAS_LIBRARIES}")
# set_property(TARGET example PROPERTY CXX_STANDARD 17)

# add project executable
set(SOURCES
    main.cpp
    model/model.h
    model/model.cpp
    trainer/trainer.h
    trainer/trainer.cpp
    utils/fileParse.h
    utils/scores.h
    utils/scores.cpp
    utils/configParse.h
    utils/configParse.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}" "${BLAS_LIBRARIES}" OpenMP::OpenMP_CXX yaml-cpp::yaml-cpp)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)