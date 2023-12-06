# notes regarding combBLAS build:
# unary_function and binary_function are no longer provided in C++17 and newer Standard modes. 
# They can be re-enabled with _LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION.
# ref: https://github.com/facebook/react-native/issues/37748

# The <parallel> header (GNU) and the <omp.h> header are associated with different parallelization 
# mechanisms. LLVM seems not to be spporting the parallel header. However, we must use LLVM (?) on
# MacOS because otherwise we cannot use OpenMP. 

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(torchtest)

# require c++14, as in CombBLAS
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS OFF)

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

# find yaml-cpp
find_package(yaml-cpp REQUIRED)
message(STATUS  "${YAML_CPP_INCLUDE_DIR}")

# find Eigen
find_package(Eigen3 REQUIRED)

# find CombBLAS
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_LIST_DIR}/libs/CombBLAS")
find_package(CombBLAS REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CombBLAS_CXX_FLAGS}")


# find OpenMP (dirty workaround for MacOS)
set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/Cellar/gcc/13.2.0/include")
set(OpenMP_C_LIB_NAMES "omp")
set(OpenMP_C "${CMAKE_C_COMPILER}")
set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/Cellar/gcc/13.2.0/include")
set(OpenMP_CXX_LIB_NAMES "omp")
set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
set(OpenMP_omp_LIBRARY "/usr/local/Cellar/libomp/17.0.2/lib/libomp.dylib")
set(CMAKE_C_COMPILER "/usr/local/Cellar/gcc/13.2.0/bin/gcc-13")
set(CMAKE_CXX_COMPILER "/usr/local/Cellar/gcc/13.2.0/bin/g++-13")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
include_directories("/usr/local/Cellar/libomp/17.0.2/include")
find_package(OpenMP REQUIRED)



set(SOURCES
    src/main.cpp
    src/model/model.h
    src/model/model.cpp
    src/trainer/trainer.h
    src/trainer/trainer.cpp
    src/utils/fileParse.h
    src/utils/scores.h
    src/utils/scores.cpp
    src/utils/configParse.h
    src/utils/configParse.cpp
)

add_executable(example src/example.cpp)
target_link_libraries(example OpenMP::OpenMP_CXX CombBLAS::CombBLAS "${TORCH_LIBRARIES}")


# add_executable(${PROJECT_NAME} ${SOURCES})
# target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}" "${BLAS_LIBRARIES}" yaml-cpp::yaml-cpp)
# set_property(TARGET example PROPERTY CXX_STANDARD 17)


