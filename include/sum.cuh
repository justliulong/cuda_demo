#ifndef CUDA_SUM
#define CUDA_SUM

#include <cuda_runtime.h>

// 全局内存上进行规约
__global__ void sum1(int* in, int* out, int in_size) {
    // int thread_id_in_block = threadIdx.x;
    int thread_id_in_grid = blockIdx.x * blockDim.x + threadIdx.x;  // 线程ID

    // 每个线程处理一个元素
    if (thread_id_in_grid < in_size) {
        // 直接在全局内存中进行规约（非常低效）
        // 这种方式实际上不可行，因为规约不能像加法一样，规约需要前面的结果合并到当前结果
        // 只能每个block处理一部分，然后在CPU上合并
    }

    int sum = 0;  // 每个block的线程有自己的sum，没有 __shared__ 修饰，不是共享内存
    for (int i = thread_id_in_grid; i < in_size; i += blockDim.x * gridDim.x) {
        sum += in[i];
    }
    
    atomicAdd(out, sum);  // 使用原子操作，将每个block的sum加到out上
}

// 引入共享内存，并且充分调用线程
__global__ void sum2(int* in, int* out, int in_size) {
    int thread_id_in_block = threadIdx.x;
    int thread_id_in_grid = blockIdx.x * blockDim.x + thread_id_in_block;  // 线程ID

    __shared__ int share_data[256];  // 静态共享内存，每个block有自己的共享内存，仅块内可见，并且至少为blockDim.x大小（为每个线程腾出位置）

    // 为了充分利用线程，这里采用 每个线程处理多个元素的方式（grid-stride）
    int sum = 0;
    for (int i = thread_id_in_grid; i < in_size; i += blockDim.x * gridDim.x) {
        sum = sum + in[i];
    }
    share_data[thread_id_in_block] = sum;  // 每个线程将结果写入共享内存
    __syncthreads();                       // 等待所有线程完成共享内存的写入，这个同步是共享内存中一旦出现修改就必须要用的同步，因为对于共享内存的操作也是异步的

    // 每个block的线程将共享内存中的结果写入全局内存
    /*
    $ 下面这段代码并不是最高效的，线程的活跃数量会越来越少：
    e.g 假设初始数据为：[1, 2, 3, 4, 5, 6, 7, 8]
    第0轮：
    ```cpp
    条件: tid % (2*1) == 0，即 tid % 2 == 0
    活跃线程: 0, 2, 4, 6
    线程0: sdata[0] += sdata[0+1] = sdata[0] + sdata[1] = 1 + 2 = 3
    线程2: sdata[2] += sdata[2+1] = sdata[2] + sdata[3] = 3 + 4 = 7
    线程4: sdata[4] += sdata[4+1] = sdata[4] + sdata[5] = 5 + 6 = 11
    线程6: sdata[6] += sdata[6+1] = sdata[6] + sdata[7] = 7 + 8 = 15
    结果: [3, 2, 7, 4, 11, 6, 15, 8]
    ```

    第1轮 (s=2):
    ```cpp
    条件: tid % (2*2) == 0，即 tid % 4 == 0
    活跃线程: 0, 4
    线程0: sdata[0] += sdata[0+2] = sdata[0] + sdata[2] = 3 + 7 = 10
    线程4: sdata[4] += sdata[4+2] = sdata[4] + sdata[6] = 11 + 15 = 26
    结果: [10, 2, 7, 4, 26, 6, 15, 8]
    ```

    第2轮 (s=4):
    ```cpp
    条件: tid % (2*4) == 0，即 tid % 8 == 0
    活跃线程: 0
    线程0: sdata[0] += sdata[0+4] = sdata[0] + sdata[4] = 10 + 26 = 36
    结果: [36, 2, 7, 4, 26, 6, 15, 8]
    ```
    此外因为执行操作的问题，下面的代码会出现线程发散的问题，因为所有线程不会进入同一个分支，导致性能下降；此外这个写法还有bank conflict 的问题

    */
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (thread_id_in_block % (2 * s) == 0) {
            share_data[thread_id_in_block] += share_data[thread_id_in_block + s];
        }
        __syncthreads();
    }

    if (thread_id_in_block == 0)
        out[blockIdx.x] = share_data[0];
}

__global__ void sum3(int* in, int* out, int in_size) {
    int thread_id_in_block = threadIdx.x;
    int thread_id_in_grid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int share_data[256];

    int sum = 0;
    for (int i = thread_id_in_grid; i < in_size; i += blockDim.x * gridDim.x) {
        sum += in[i];
    }

    share_data[thread_id_in_block] = sum;
    __syncthreads();

    // 反向遍历，这样能式线程访问的bank相隔远，减少bank conflict
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (thread_id_in_block < s) {
            share_data[thread_id_in_block] += share_data[thread_id_in_block + s];
        }
        __syncthreads();
    }

    if (thread_id_in_block == 0)
        out[blockIdx.x] = share_data[0];
}


// 模板特化，warp内部天然支持同步，就避免了 __syncthreads() 的时间问题
template <const int blockSize>
__device__ __forceinline__ void warpReduce(volatile int* sdata, int tid) { // 这里要注意，volatile 是为了防止编译器优化，因为编译器可能会将 __syncthreads() 之前的代码优化掉，导致结果不正确
    if (blockSize >= 64) sdata[tid] = sdata[tid] + sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] = sdata[tid] + sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] = sdata[tid] + sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] = sdata[tid] + sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] = sdata[tid] + sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] = sdata[tid] + sdata[tid + 1];
};

template <const int blockSize>
__global__ void sum4(int* in, int* out, int n) {
    int thread_id_in_block = threadIdx.x;
    int thread_id_in_grid = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ int share_data[];

    int sum = 0;
    for (int i = thread_id_in_grid; i < n; i += blockDim.x * gridDim.x) {
        sum += in[i];
    }

    share_data[thread_id_in_block] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (thread_id_in_block < s) {
            share_data[thread_id_in_block] += share_data[thread_id_in_block + s];
        }
        __syncthreads();
    }
    
    if (thread_id_in_block < 32) {
        warpReduce<blockSize>(share_data, thread_id_in_block); // 由于使用的是动态共享内存，所以并不方便将share_data的数据转移到同一个warp中用warp shuffle进行规约，应为共享内存的大小受blockSize影响
    }

    if (thread_id_in_block == 0)
        out[blockIdx.x] = share_data[0];
}

// warp shuffle 优化
__global__ void sum5(int* in, int* out, int n) {
    int thread_id_in_block = threadIdx.x;
    int thread_id_in_grid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = thread_id_in_block & 31; // equal to thread_id_in_block % 32
    int warp_id = thread_id_in_block >> 5; // equal to thread_id_in_block / 32


    int sum = 0;
    for (int i = thread_id_in_grid; i < n; i += blockDim.x * gridDim.x) {
        sum += in[i];
    }

    // 将warp内的数据进行规约，后面再考虑将一个block中的所有warp求和
    for(int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // 进行block内warp的求和，每个 warp 中的 lane0 存储了对应个warp的和
    __shared__ int warp_results[32]; // 一个block中最多有32个warp
    if(lane_id == 0)
        warp_results[warp_id] = sum; // 将warp内的和存储到共享内存中
    __syncthreads();
    
    if (warp_id == 0) {
        int warp_sum = (lane_id < (blockDim.x / warpSize)) ? warp_results[lane_id] : 0.0f; //@ 由于 warp 的数量不一定能达到32,最后规约的时候只需要 num of warp 个线程参数，所以在这里进行设置, 多余的线程设置为0,这样就能统一进行规约了
        
        // 只有前几个线程（线程数=warps数）参与规约
        if (lane_id < (blockDim.x / warpSize)) {
           for (int offset = warpSize/2; offset > 0; offset /= 2) {
               // 获取下方offset处的值
                int other = __shfl_down_sync(0xffffffff, warp_sum, offset);
                
                // 仅当在有效范围内时累加
                if (lane_id < offset) {
                    warp_sum += other;
                }
           }
        if (lane_id == 0)
            out[blockIdx.x] = warp_sum;
       }

    }
    
}

#endif 