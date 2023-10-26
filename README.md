# Distributed-THNN

GNN Guide - https://distill.pub/2021/gnn-intro/


### How to run the porject on Linux

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
