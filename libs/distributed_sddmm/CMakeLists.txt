cmake_minimum_required(VERSION 3.16)

project(distributed_sddmmm LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})

find_package(MPI REQUIRED)
find_package(BLAS REQUIRED)
find_package(OpenMP REQUIRED)

# These link to the copy of CombBLAS in DiBELLA
# set(CombBLAS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs/CombBLAS)
# set(CombBLAS_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/libs/CombBLAS)

# # Add the library as a subdirectory
# add_subdirectory(${CombBLAS_SOURCE_DIR} ${CombBLAS_BINARY_DIR})

# # Add include directories
# include_directories(BEFORE SYSTEM ${CombBLAS_SOURCE_DIR} ${CombBLAS_BINARY_DIR})

message(STATUS "${CombBLAS_SOURCE_DIR} ${CombBLAS_BINARY_DIR}")

message(STATUS "cmake prefix path: ${CMAKE_PREFIX_PATH}")

# Link to Eigen
set(EIGEN_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/libs/eigen)

message(STATUS "Eigen include: ${EIGEN_INCLUDE_DIR}")
include_directories(${EIGEN_INCLUDE_DIR})
# find_package (Eigen3 3.3 REQUIRED NO_MODULE)

#add_executable(bench_er bench_erdos_renyi.cpp benchmark_dist.cpp sparse_kernels.cpp common.cpp als_conjugate_gradients.cpp) 
# add_executable(bench_file bench_file.cpp benchmark_dist.cpp sparse_kernels.cpp common.cpp als_conjugate_gradients.cpp) 
# add_executable(bench_heatmap bench_heatmap.cpp benchmark_dist.cpp sparse_kernels.cpp common.cpp als_conjugate_gradients.cpp) 
# add_executable(local_kernel_benchmark sparse_kernels.cpp local_kernel_benchmark.cpp common.cpp)
# add_executable(scratch scratch.cpp sparse_kernels.cpp common.cpp als_conjugate_gradients.cpp)
# add_executable(petsc_bench petsc_baseline/spmm_test.cpp) 
# add_executable(random_permute random_permute.cpp) 

#add_executable(test_async test_async_strategies.cpp common.cpp) 
#add_executable(par_iotest ParIOTest.cpp) 

#foreach(TARGET bench_er) #bench_file bench_heatmap local_kernel_benchmark scratch)
#    target_compile_options(${TARGET} PRIVATE -O3 -march=knl -DMKL_ILP64  -m64  -I$ENV{MKLROOT}/include) 
#    target_link_options(${TARGET} PRIVATE -L$ENV{MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl)
#    target_link_libraries(${TARGET} PRIVATE MPI::MPI_CXX ${BLAS_LIBRARIES} OpenMP::OpenMP_CXX CombBLAS GraphGenlib Eigen3::Eigen)
#    target_precompile_headers(${TARGET} PRIVATE json.hpp <Eigen/Dense>)
#endforeach()


# foreach(TARGET random_permute)
#    target_compile_options(${TARGET} PRIVATE -O3 -march=haswell -DMKL_ILP64  -m64  -I$ENV{MKLROOT}/include) 
#    target_link_options(${TARGET} PRIVATE -L$ENV{MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl)
#    target_link_libraries(${TARGET} PRIVATE MPI::MPI_CXX ${BLAS_LIBRARIES} OpenMP::OpenMP_CXX CombBLAS GraphGenlib )
#    target_precompile_headers(${TARGET} PRIVATE json.hpp <Eigen/Dense>)
# endforeach()


#foreach(TARGET par_iotest)
#    target_compile_options(${TARGET} PRIVATE -O3 -march=haswell -DMKL_ILP64  -m64  -I$ENV{MKLROOT}/include) 
#    target_link_options(${TARGET} PRIVATE -L$ENV{MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl)
#    target_link_libraries(${TARGET} PRIVATE MPI::MPI_CXX ${BLAS_LIBRARIES} OpenMP::OpenMP_CXX CombBLAS GraphGenlib )
#    target_precompile_headers(${TARGET} PRIVATE json.hpp <Eigen/Dense>)
#endforeach()

add_executable(test main.cpp)
message(STATUS "MKL_LIBRARIES: ${MKL_LIBRARIES}")
target_include_directories(test PRIVATE $ENV{MKLROOT}/include)
target_compile_options(test PRIVATE -O3 -march=knl -DMKL_ILP64  -m64  -I$ENV{MKLROOT}/include) 
target_link_options(test PRIVATE -L$ENV{MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl)
target_link_libraries(test PRIVATE MPI::MPI_CXX ${BLAS_LIBRARIES} OpenMP::OpenMP_CXX CombBLAS GraphGenlib PUBLIC MKL::MKL)

# target_link_libraries(test PRIVATE MPI::MPI_CXX ${BLAS_LIBRARIES} OpenMP::OpenMP_CXX CombBLAS GraphGenlib Eigen3::Eigen)
target_precompile_headers(test PRIVATE json.hpp <Eigen/Dense>)
