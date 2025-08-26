
<div id="top" align="center">

# CUDA DEMO
  A repository for learning and testing CUDA programming </br>

  [![CUDA](https://img.shields.io/badge/CUDA-12.9-blue.svg)](https://developer.nvidia.com/cuda-toolkit)
  [![Compute Capability](https://img.shields.io/badge/Compute%20Capability-3.0+-blue.svg)](https://developer.nvidia.com/cuda-gpus)

</div>

This project is built and tested under the EndeavorOS(arch series) operating system. The graphics card driver needs to be installed and compiled according to the corresponding CUDA version:

## NVIDIA Driver
```shell
yay -S nvidia-dkms nvidia-utils nvidia-settings
```

## CUDA
```shell
yay -S cuda
```
It is usually installed together with cudnn
```shell
yay -S cudnn
```
Check if the installation was successful：

```shell
nvcc -V

# The output is similar to the following
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Tue_May_27_02:21:03_PDT_2025
Cuda compilation tools, release 12.9, V12.9.86
Build cuda_12.9.r12.9/compiler.36037853_0

```

## compile
```shell
cmake -B build

cmake --build build --parallel 8
```

## run
```shell
./bin/main
```