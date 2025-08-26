#ifndef ADD_CUH
#define ADD_CUH

#include <cuda_runtime.h>

__global__ void add(int *arr, int *arr2, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        // arr[i] += arr2[i]; //$ 这种写法有潜在的数据竞争问题，这里没有明确的读-计算-写模式
        arr[i] = arr[i] + arr2[i];  // @ 这种写法就是读-计算-写模式，不会有潜在的数据竞争问题
    }
}

//? 通过测试发现，add 中如果是大型矩阵参与运算，if的性能就不会比for更好
// __global__ void add(int *arr, int *arr2, int n) {
//     int id = blockIdx.x * blockDim.x + threadIdx.x;
//     if (id < n) {
//         arr[id] += arr2[id];
//     }
// }

#endif  // ADD_CUH