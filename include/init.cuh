#ifndef INIT_CUH
#define INIT_CUH

#include <cuda_runtime.h>

// 核函数，这种函数里面只能调用设备函数，他自己都没有部分调用其他核函数
__global__ void init_array(int *arr, int n) {
    // 网格扁平化 blockIdx.x * blockDim.x + threadIdx.x 表示当前线程在全局线程索引中的位置， 每次循环加 blockDim.x * gridDim.x 表示每次循环增加的线程数
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {  //? 这个 i 没有修饰符，就是创建在寄存器上，只有线程内部可见
        arr[i] = i;
        // std::cout << std::format("{} \n", i);  //! 这里不正确，这个format和cout都是标准库里面给cpu使用的主机函数，不能在核函数中调用
    }
}
#endif