# Distributed-THNN

GNN Guide - https://distill.pub/2021/gnn-intro/


### How to run the project on Linux

Install BLAS library:
```
sudo apt install libopenblas-dev
```

Move to the root folder of the git repository:
```
cd <path-to-repo>/distributed-THNN
```

Create a libs and a build folder:
```
mkdir build && mkdir libs
```

Install Libtorch (Version without CUDA):
```
cd libs
```
```
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
```
```
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
```

Finally do the following commands:
```
cd ../build
```
```
cmake ..
```

```
make
```

Now you can run the executable files that have been produced. 


### How to run the project on MACOS

```
brew install openblas
```
Download torchlib and install to lib/torchlib:
```
 wget https://github.com/mlverse/libtorch-mac-m1/releases/download/LibTorch/libtorch-v2.1.0.zip && unzip libtorch-v2.1.0.zip -d libs/torchlib
```

Create build directory and executables:
```
mkdir build
cd build
cmake ..
make
```
