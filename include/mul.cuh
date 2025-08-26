#ifndef CUDA_MATRIX_MUL_CUH
#define CUDA_MATRIX_MUL_CUH

#include <cuda_runtime.h>

template <typename T>
__global__ void mul1(T *a, T *b, T *c, int M, int N, int K) {
    // M * N @ N * K -> M * K
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 块内线程行编号，也是结果的行编号
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 块内线程列编号，也是结果的列编号
    T sum = 0;
    if (row < M && col < K) {
        #pragma unroll
        for (int i = 0; i < N; i++) {
            // A: row行, i列； B: i行, col列
            sum += a[row * N + i] * b[i * K + col];  // a的第row行的第i个元素 * b的第i列的第col个元素
        }
    }
    if (row < M && col < K)
        c[row * K + col] = sum;
}

template <typename T>
__global__ void mul_3d1(T *a, T *b, T *c, int B, int M, int N, int K) {
    // B * M * N @ B * N * K -> B * M * K
    int batch = blockIdx.z * blockDim.z + threadIdx.z;  // 块内线程批次编号，也是结果的批次编号
    int row = blockIdx.y * blockDim.y + threadIdx.y;    // 块内线程行编号，也是结果的行编号
    int col = blockIdx.x * blockDim.x + threadIdx.x;    // 块内线程列编号，也是结果的列编号

    // 按照行优先存储，对于 C 矩阵 第batch个矩阵的第 row 行第 col 列元素：batch * M * K + row * K + col

    // 对于 A(M*N) 矩阵 第batch个矩阵的第 { r } 行第 { c } 列元素：batch * M * N + r * N + c
    T sum = 0;
    if (batch < B && row < M && col < K) {
        #pragma unroll
        for (int i = 0; i < N; i++) {
            // A: row行, i列； B: i行, col列
            sum += a[batch * M * N + row * N + i] * b[batch * N * K + i * K + col];  // a的第row行的第i个元素 * b的第i列的第col个元素
        }
    }
    if (batch < B && row < M && col < K)
    c[batch * M * K + row * K + col] = sum;
}

// 使用共享内存的分块技术，将矩阵分块，每个线程计算一个块内的乘法结果
template <typename T, const int TILE_SIZE = 16>
__global__ void mul2(T *a, T *b, T *c, int M, int N, int K) {
    // M * N @ N * K -> M * K
    __shared__ T ds_a[TILE_SIZE][TILE_SIZE];  // 共享内存，用于存储A矩阵的块
    __shared__ T ds_b[TILE_SIZE][TILE_SIZE];  // 共享内存，用于存储B矩阵的块

    int bx = blockIdx.x, by = blockIdx.y;    // 块编号
    int tx = threadIdx.x, ty = threadIdx.y;  // 线程编号

    // 在这里 TILE_SIZE == blockDim.x == blockDim.y
    int row = by * TILE_SIZE + ty;  // 线程结果对应的行编号
    int col = bx * TILE_SIZE + tx;  // 线程结果对应的列编号

    T sum = 0;  // 本线程计算结果，也就是结果中第 row 行 第 col 列的元素
    // 先将数据加载进入共享内存
    for (int i = 0; i < (N + TILE_SIZE - 1) / TILE_SIZE; ++i) {  // 由于 TILE_SIZE 一般是覆盖不住矩阵的全部值，所以会反复计算几次，这里也符合矩阵的分开加载结果
        int a_col = i * TILE_SIZE + tx;                          // A 是固定行，变化列
        int b_row = i * TILE_SIZE + ty;                          // B 是固定列，变化行

        // 加载 A 的值
        if (row < M && a_col < N) {
            ds_a[ty][tx] = a[row * N + a_col];
        } else {
            ds_a[ty][tx] = T(0);
        }

        // 加载 B 的值
        if (b_row < N && col < K) {
            ds_b[ty][tx] = b[b_row * K + col];
        } else {
            ds_b[ty][tx] = T(0);
        }

        __syncthreads();  // 同步，确保共享内存中的数据已经加载完成

        // 计算结果
        for (int j = 0; j < TILE_SIZE; ++j) {
            sum += ds_a[ty][j] * ds_b[j][tx]; // 理论上也能直接在全局内存上同步c[row * K + col] += ds_a[ty][j] * ds_b[j][tx]; ，但是这样需要反复访问全局内存，速度很低
        }

        __syncthreads();  // 同步，确保计算完成
    }

    if (row < M && col < K) {
        c[row * K + col] = sum;
    }
}

template <typename T, const int TILE_SIZE = 16>
__global__ void mul_3d2(T *a, T *b, T *c, int B, int M, int N, int K) {
    // B * M * N @ B * N * K -> B * M * K
    __shared__ T ds_a[TILE_SIZE][TILE_SIZE];  // 共享内存，用于存储A矩阵的块
    __shared__ T ds_b[TILE_SIZE][TILE_SIZE];  // 共享内存，用于存储B矩阵的块

    int batch = blockIdx.z ;  // 块内线程批次编号，也是结果的批次编号
    if(batch >= B) return; // 防止越界
    int bx = blockIdx.x, by = blockIdx.y;               // 块编号
    int tx = threadIdx.x, ty = threadIdx.y;             // 线程编号

    // 每个线程只会处理一批，不同批在不同的block上进行计算
    int row = by * TILE_SIZE + ty;  // 线程结果对应的行编号
    int col = bx * TILE_SIZE + tx;  // 线程结果对应的列编号

    T sum = 0;  // 本线程计算结果，也就是结果中第 row 行 第 col 列的元素
    // 先将数据加载进入共享内存
    for (int i = 0; i < (N + TILE_SIZE - 1) / TILE_SIZE; ++i) {  // 由于 TILE_SIZE 一般是覆盖不住矩阵的全部值，所以会反复计算几次，这里也符合矩阵的分开加载结果
        int a_col = i * TILE_SIZE + tx;                          // A 是固定行，变化列
        int b_row = i * TILE_SIZE + ty;                          // B 是固定列，变化行

        // 加载 A 的值
        if ( row < M && a_col < N) {
            ds_a[ty][tx] = a[batch * M * N + row * N + a_col];
        } else {
            ds_a[ty][tx] = T(0);
        }

        // 加载 B 的值
        if (b_row < N && col < K) {
            ds_b[ty][tx] = b[batch * N * K + b_row * K + col];
        } else {
            ds_b[ty][tx] = T(0);
        }

        __syncthreads();  // 同步，确保共享内存中的数据已经加载完成

        // 计算结果
        for (int j = 0; j < TILE_SIZE; ++j) {
            sum += ds_a[ty][j] * ds_b[j][tx];
        }
        
        __syncthreads();  // 同步，确保计算完成
    }

    if (batch < B && row < M && col < K) 
        c[batch * M * K + row * K + col] = sum;
}

// 每个线程处理多batch
template <typename T, const int TILE_SIZE = 16, const int BATCH_SIZE = 4>
__global__ void mul_3d3(T *a, T *b, T *c, int B, int M, int N, int K) {
    // B * M * N @ B * N * K -> B * M * K

}
#endif  // !CUDA_MATRIX_MUL_CUH