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

set(OpenGL_GL_PREFERENCE "GLVND")

find_package(BLAS REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BLAS_CXX_FLAGS}")

# add_compile_options("-pthread")
# set(CMAKE_THREAD_LIBS_INIT "-lpthread")
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
# set(CMAKE_HAVE_THREADS_LIBRARY 1)
# set(CMAKE_USE_WIN32_THREADS_INIT 0)
# set(CMAKE_USE_PTHREADS_INIT 1)
# set(THREADS_PREFER_PTHREAD_FLAG ON)

include(FetchContent)
FetchContent_Declare(
		yaml-cpp
		GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
		GIT_SHALLOW	ON
		GIT_TAG 0.8.0
	)
FetchContent_MakeAvailable(yaml-cpp)


include_directories(BEFORE SYSTEM ${yaml-cpp_SOURCE_DIR} ${yaml-cpp_BINARY_DIR}/include)

# include(FetchContent)
# FetchContent_Declare(
#         CombBLAS
#         GIT_REPOSITORY https://github.com/PASSIONLab/CombBLAS.git
#         # GIT_TAG v2.0.0
#     )
# FetchContent_MakeAvailable(CombBLAS)
# message(STATUS "combblas source dir: ${combblas_SOURCE_DIR}")
# message(STATUS "combblas binary dir: ${combblas_BINARY_DIR}")
# include_directories(BEFORE SYSTEM ${CombBLAS_SOURCE_DIR} ${CombBLAS_BINARY_DIR})

set(CombBLAS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs/CombBLAS)
set(CombBLAS_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/libs/CombBLAS)

# Add the library as a subdirectory
add_subdirectory(${CombBLAS_SOURCE_DIR} ${CombBLAS_BINARY_DIR})

# Add include directories
include_directories(BEFORE SYSTEM ${CombBLAS_SOURCE_DIR} ${CombBLAS_BINARY_DIR})

# set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_LIST_DIR}/libs/eigen")
# find_package(Eigen3 REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EIGEN3_CXX_FLAGS}")

include(FetchContent)
FetchContent_Declare(
        fast_matrix_market
        GIT_REPOSITORY https://github.com/alugowski/fast_matrix_market
        GIT_TAG main
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(fast_matrix_market)

set(SOURCES
    src/main.cpp
    src/model/model.h
    src/model/model.cpp
    src/model/dist-model.h
    src/model/dist-model.cpp
    src/trainer/trainer.h
    src/trainer/trainer.cpp
    src/utils/fileParse.h
    src/utils/scores.h
    src/utils/scores.cpp
    src/utils/configParse.h
    src/utils/configParse.cpp
    src/utils/parDenseGEMM.h
    src/utils/parDenseGEMM.cpp
    src/utils/DenseMatrix.h
)

add_executable(example src/example.cpp)
target_link_libraries(example "${TORCH_LIBRARIES}" "${BLAS_LIBRARIES}" yaml-cpp CombBLAS "${EIGEN3_LIBRARIES}")
set_property(TARGET example PROPERTY CXX_STANDARD 17)

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}" "${BLAS_LIBRARIES}" yaml-cpp CombBLAS "${EIGEN3_LIBRARIES}")
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

add_executable(example_2 src/example_2.cpp)
target_link_libraries(example_2 "${TORCH_LIBRARIES}" "${BLAS_LIBRARIES}" yaml-cpp CombBLAS "${EIGEN3_LIBRARIES}")
set_property(TARGET example PROPERTY CXX_STANDARD 17)

add_executable(openBLAS_example src/openBLAS_example.cpp src/utils/parDenseGEMM.h src/utils/parDenseGEMM.cpp src/utils/DenseMatrix.h)
target_link_libraries(openBLAS_example "${TORCH_LIBRARIES}" "${BLAS_LIBRARIES}" yaml-cpp CombBLAS "${EIGEN3_LIBRARIES}" fast_matrix_market::fast_matrix_market)
set_property(TARGET example PROPERTY CXX_STANDARD 17)