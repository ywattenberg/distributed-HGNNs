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
FetchContent_GetProperties(yaml-cpp)
if(NOT yaml-cpp_POPULATED)
	message(STATUS "Populating yaml-cpp...")
	FetchContent_Populate(yaml-cpp)
	# Add here options for yaml-cpp building
    set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "disable yaml tests")
	add_subdirectory(${yaml-cpp_SOURCE_DIR} ${yaml-cpp_BINARY_DIR})
	message(STATUS "Done.")
endif()
include_directories(BEFORE SYSTEM ${yaml-cpp_SOURCE_DIR} ${yaml-cpp_BINARY_DIR}/include)

include(FetchContent)
FetchContent_Declare(
        CombBLAS
        GIT_REPOSITORY https://github.com/PASSIONLab/CombBLAS.git
        # GIT_TAG v2.0.0
    )
FetchContent_GetProperties(CombBLAS)
if(NOT CombBLAS_POPULATED)
    message(STATUS "Populating combblas...")
    FetchContent_Populate(CombBLAS)
    # Add here options for combblas building
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    add_subdirectory(${CombBLAS_SOURCE_DIR} ${CombBLAS_BINARY_DIR})
    message(STATUS "Done.")
endif()
message(STATUS "combblas source dir: ${combblas_SOURCE_DIR}")
message(STATUS "combblas binary dir: ${combblas_BINARY_DIR}")
include_directories(BEFORE SYSTEM ${CombBLAS_SOURCE_DIR} ${CombBLAS_BINARY_DIR}/include)

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

# add_executable(example example.cpp)
# target_link_libraries(example "${TORCH_LIBRARIES}" "${BLAS_LIBRARIES}" yaml-cpp CombBLAS)
# set_property(TARGET example PROPERTY CXX_STANDARD 17)

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}" "${BLAS_LIBRARIES}" yaml-cpp CombBLAS)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)