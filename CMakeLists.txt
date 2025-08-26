cmake_minimum_required( VERSION 3.24) # 对于CUDA编程，至少需要CMake 3.18

set(CMAKE_CXX_STANDARD 20) # 在 RTX900 系列显卡以上的  nvcc 编译器是支持到了 c++17 版本，越高支持的约好 目前的 nvcc 的 12.9 已经支持到了 c++20 ，不过暂时没有任何c++23 的支持

set(CMAKE_BUILD_TYPE Release) # 设置编译类型为 Release, 默认开启 O3 优化

project(cuda_demo LANGUAGES CXX CUDA) # 设置项目名称为 cuda_demo, 语言使用 C++ 和 CUDA

# set(CMAKE_CUDA_ARCHITECTURES 50)

# set(CMAKE_CUDA_COMPILER "/opt/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES "native")  # 自动检测GPU架构，这样会使你编译的代码只能在你目前编译的机器上运行，到其他架构的显卡上结果可能就会出现问题，不然就再次编译，不然就制定多个架构类型

message(STATUS "Final CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# 设置生成的目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin) # 设置可执行文件输出目录

file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

file(GLOB_RECURSE CUDA_FILES "${CMAKE_SOURCE_DIR}/src/*.cu") # 查找所有 .cu 文件

add_executable(main src/main.cu ) # 添加可执行文件 main.cpp

target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/include) # 添加头文件目录