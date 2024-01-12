# Distributed-HGNNs
We implement a distributed-memory full-batch version of "Hypergraph Neural Networks" by Feng et al..
This makes it possible to train on larger graph datasets without using mini-batch techniques, which can introduce additional biases and often cost some precision 
We show that our implementation achieves the same accuracy as the original HGNN model. Further, we compare runtimes to the highly optimized and non-distributed model implemented using the torch framework.

### How to run the project on Linux

Clone the repository using git clone. As our repository has a submodule you have to run the following commands after you cloned it:
```
git submodule init
```
and
```
git pull --recurse-submodules
```


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


### Train the HGNN Model
The data needed for the training is mostly present in the folder "data/m_g_ms_gs". The features are too big for the repository. You can download here form our Drive-Folder:
https://drive.google.com/drive/folders/18Bhjj8Mbt2g_JBRQB8JpzUG6BzTkGr6c?usp=sharing. 

It's best to also move the features also into the "data/m_g_ms_gs" folder. 

The file example_model.cpp runs the HGNN-Model with our distributed implementation. After having executed the steps above, it can be run with the following command:
```
mpiexec -np 4 <path-to-directory>/distributed-HGNNs/build/dist-hgnn -c "<path-to-directory>/distributed-HGNNs/config/dist-model.yaml" -d "<path-to-directory>/distributed-HGNNs/" -i 1 -t 1
```
Depending on the machine, it may be necessary to set the number of omp-threads manually to a lower number.
```
export OMP_NUM_THREADS=1
```
Further replace <path-to-directory> with the path where you stored this git repository. This command will train the model with our distributed algorithm using 4 MPI-Processes (declared by -np 4). Please notice, that it can only be started using a squared number of MPI-Processes/Nodes, i.e. 4,9,16,... . If t is set to 1, it also measures the time an epoch takes. 
You can also create other config files or change the parameters in the "dist-model.yaml"-file to play around with the hyperparameters. 

The (non-distributed) torch implementation can be run with the following command:
```
<path-to-directory>/distributed-HGNNs/build/dist-hgnn -c "$<path-to-directory>/distributed-HGNNs/config/torch-model.yaml" -d "<path-to-directory>/distributed-HGNNs/" -i 1 -t 1
```

We also provide scripts (in the scripts folder) to run our model and some benchmarks on euler using slurms. However, you have to first install libtorch and BLAS on euler in order to use them. 








