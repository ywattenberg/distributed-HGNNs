cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(dist-hgnn)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(MACOSX TRUE)
endif()


message(STATUS "${CMAKE_CURRENT_LIST_DIR}")

# find libtorch
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_LIST_DIR}/libs/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_LIST_DIR}/libs/mkl")
# find_package(MKL REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MKL_CXX_FLAGS}")

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

set(EIGEN_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/libs/eigen)


# set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_LIST_DIR}/libs/eigen")
# find_package(Eigen3 REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EIGEN3_CXX_FLAGS}")

# set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_LIST_DIR}/libs/distributed_sddmm")
# # find_package(Eigen3 REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EIGEN3_CXX_FLAGS}")

set(distributed_sddmm_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs/distributed_sddmm)
set(distributed_sddmm_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/libs/distributed_sddmm)

# Add the library as a subdirectory
add_subdirectory(${distributed_sddmm_SOURCE_DIR} ${distributed_sddmm_BINARY_DIR})

# Add include directories
include_directories(BEFORE SYSTEM ${distributed_sddmm_SOURCE_DIR} ${distributed_sddmm_BINARY_DIR})

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
    src/trainer/trainer.h
    src/trainer/trainer.cpp
    src/utils/fileParse.h
    src/utils/scores.h
    src/utils/scores.cpp
    src/utils/configParse.h
    src/utils/configParse.cpp
    src/utils/parDenseGEMM.h
    src/utils/parDenseGEMM.cpp
    src/DenseMatrix/DenseMatrix.h
    src/DenseMatrix/DenseMatrix.cpp
    src/DenseMatrix/DenseMatrixAlgorithms.h
    src/DenseMatrix/DenseMatrixAlgorithms.cpp
    src/DenseMatrix/DistributedDGEMM.cpp
    src/DenseMatrix/DistributedSpDGEMM.cpp
    src/DenseMatrix/DenseMatrixPRead.cpp
    src/utils/LossFn.h
    src/utils/LossFn.cpp
    src/utils/DerivativeFunctions.h
    src/utils/DerivativeFunctions.cpp
    src/model/dist-model.h
    src/model/dist-model.cpp
    src/model/dist-model-w.h
    src/model/dist-model-w.cpp
    src/trainer/dist-trainer.h
    src/trainer/dist-trainer.cpp
)

set(DIST 
    src/model/dist-model.h
    src/model/dist-model.cpp
    src/utils/fileParse.h
    src/utils/configParse.h
    src/utils/configParse.cpp
    src/utils/parDenseGEMM.h
    src/utils/parDenseGEMM.cpp
    src/DenseMatrix/DenseMatrix.h
    src/DenseMatrix/DenseMatrix.cpp
    src/DenseMatrix/DenseMatrixAlgorithms.h
    src/DenseMatrix/DenseMatrixAlgorithms.cpp
    src/DenseMatrix/DistributedDGEMM.cpp
    src/DenseMatrix/DistributedSpDGEMM.cpp
    src/DenseMatrix/DenseMatrixPRead.cpp
    src/utils/LossFn.h
    src/utils/LossFn.cpp
    src/utils/DerivativeFunctions.h
    src/utils/DerivativeFunctions.cpp
    src/model/dist-model.h
    src/model/dist-model.cpp
    src/model/dist-model-w.h
    src/model/dist-model-w.cpp
)

add_executable(dist-model-example src/examples/example_model.cpp ${DIST})
target_link_libraries(dist-model-example "${TORCH_LIBRARIES}" "${BLAS_LIBRARIES}" yaml-cpp CombBLAS)
set_property(TARGET dist-model-example PROPERTY CXX_STANDARD 17)

add_executable(example src/examples/example.cpp)
target_link_libraries(example "${TORCH_LIBRARIES}" "${BLAS_LIBRARIES}" yaml-cpp CombBLAS "${EIGEN3_LIBRARIES}")
set_property(TARGET example PROPERTY CXX_STANDARD 17)

# add_executable(example_2 src/examples/example_2.cpp ${SOURCES} ${DIST})
# target_link_libraries(example_2 "${TORCH_LIBRARIES}" "${BLAS_LIBRARIES}" yaml-cpp CombBLAS "${EIGEN3_LIBRARIES}")
# target_include_directories(example_2 PRIVATE ${EIGEN_INCLUDE_DIR})
# set_property(TARGET example_2 PROPERTY CXX_STANDARD 17)

# add_executable(openBLAS_example src/examples/openBLAS_example.cpp ${SOURCES} ${DIST})
# target_link_libraries(openBLAS_example "${TORCH_LIBRARIES}" "${BLAS_LIBRARIES}" yaml-cpp CombBLAS "${EIGEN3_LIBRARIES}" fast_matrix_market::fast_matrix_market)
# set_property(TARGET openBLAS_example PROPERTY CXX_STANDARD 17)

add_executable(${PROJECT_NAME} ${SOURCES} ${DIST})
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}" "${BLAS_LIBRARIES}" yaml-cpp CombBLAS "${EIGEN3_LIBRARIES}")
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

add_executable(example_sddmm src/examples/example_sddmm.cpp)
target_link_libraries(example_sddmm "${TORCH_LIBRARIES}" "${BLAS_LIBRARIES}" yaml-cpp CombBLAS "${EIGEN3_LIBRARIES}")
target_include_directories(example_sddmm PRIVATE ${EIGEN_INCLUDE_DIR})
set_property(TARGET example_sddmm PROPERTY CXX_STANDARD 17)


