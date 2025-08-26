// #include <cuda_runtime.h>

// #include <cstdio>
#include <chrono>
#include <format>
#include <iostream>
#include <vector>

#include "add.cuh"
#include "init.cuh"
#include "mul.cuh"
#include "sum.cuh"
// #include <thrust/*.h> // thrust库封装了cuda的很多操作，对标cpu 里面的 std 标准库

#define CUDA_CHECK(call)                                                                                \
    do {                                                                                                \
        cudaError_t err = call;                                                                         \
        if (err != cudaSuccess) {                                                                       \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                         \
        }                                                                                               \
    } while (0)

// 主机函数
__host__ void hello() {  // tip 也可以不使用__host__，因为默认就是主机函数
    std::cout << "hello host \n";
}

//! 设备函数，这种函数允许被其他设备函数和核函数调用，但是主机函数不允许调用设备函数，设备函数里面也没有办法调用主机函数
__device__ void hello_device() {
    // ...
}

// 模拟 matrix_num 个 大矩阵{B * M * N} * matrix_num 个大矩阵{B * N * K} = matrix_num 个大矩阵{B * M * K}，这里使用正常的串行计算
void MulBySerial(std::vector<int *> h_arr1, std::vector<int *> h_arr2, std::vector<int *> out, int B, int M, int N, int K, int matrix_num) {
    int a_matrix_size = B * M * N;
    int b_matrix_size = B * N * K;
    int out_matrix_size = B * M * K;
    for (int i = 0; i < matrix_num; i++) {
        // 开辟一定的空间给device上的数组
        int *d_arr1;
        int *d_arr2;
        int *res;

        CUDA_CHECK(cudaMalloc(&d_arr1, a_matrix_size * sizeof(int)));

        CUDA_CHECK(cudaMalloc(&d_arr2, b_matrix_size * sizeof(int)));

        CUDA_CHECK(cudaMalloc(&res, out_matrix_size * sizeof(int)));

        // h2d
        CUDA_CHECK(cudaMemcpy(d_arr1, h_arr1[i], a_matrix_size * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_arr2, h_arr2[i], b_matrix_size * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(res, out[i], out_matrix_size * sizeof(int), cudaMemcpyHostToDevice));

        dim3 blockSize = {16, 16};
        dim3 gridSize = {(K + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y};
        // kernel
        mul_3d2<<<gridSize, blockSize>>>(d_arr1, d_arr2, res, B, M, N, K);

        // d2h
        CUDA_CHECK(cudaMemcpy(out[i], res, out_matrix_size * sizeof(int), cudaMemcpyDeviceToHost));  // 这个地方会自动同步，就不需要 cudaDeviceSynchronize() 了

        // 释放device上的数组
        CUDA_CHECK(cudaFree(d_arr1));
        CUDA_CHECK(cudaFree(d_arr2));
        CUDA_CHECK(cudaFree(res));
    }
}

void MulByStream(std::vector<int *> h_arr1, std::vector<int *> h_arr2, std::vector<int *> out, int B, int M, int N, int K, int matrix_num, int stream_num) {
    // 准备阶段，创建流、数据存储
    int a_matrix_size = B * M * N;
    int b_matrix_size = B * N * K;
    int out_matrix_size = B * M * K;
    // 创建流
    std::vector<cudaStream_t> streams(stream_num);
    for (int i = 0; i < stream_num; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }
    // 使用vector 来管理将来用于存储在device上的数组
    std::vector<int *> d_arr1(stream_num);
    std::vector<int *> d_arr2(stream_num);
    std::vector<int *> res(stream_num);
    for (int j = 0; j < stream_num; ++j) {
        CUDA_CHECK(cudaMalloc(&d_arr1[j], a_matrix_size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_arr2[j], b_matrix_size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&res[j], out_matrix_size * sizeof(int)));
    }
    for (int i = 0; i < matrix_num; ++i) {
        // 记录当前的流id
        int stream_id = i % stream_num;

        // 异步 h2d
        cudaMemcpyAsync(d_arr1[stream_id], h_arr1[i], a_matrix_size * sizeof(int), cudaMemcpyHostToDevice, streams[stream_id]);
        cudaMemcpyAsync(d_arr2[stream_id], h_arr2[i], b_matrix_size * sizeof(int), cudaMemcpyHostToDevice, streams[stream_id]);
        cudaMemcpyAsync(res[stream_id], out[i], out_matrix_size * sizeof(int), cudaMemcpyHostToDevice, streams[stream_id]);

        dim3 blockSize = {16, 16};
        dim3 gridSize = {(K + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y};
        // kernel
        mul_3d2<<<gridSize, blockSize, 0, streams[stream_id]>>>(d_arr1[stream_id], d_arr2[stream_id], res[stream_id], B, M, N, K);

        // 异步 d2h
        cudaMemcpyAsync(out[i], res[stream_id], out_matrix_size * sizeof(int), cudaMemcpyDeviceToHost, streams[stream_id]);
    }

    for (auto &stream : streams)
        CUDA_CHECK(cudaStreamSynchronize(stream));

    // 销毁流
    for (int i = 0; i < stream_num; ++i) {
        CUDA_CHECK(cudaFree(d_arr1[i]));
        CUDA_CHECK(cudaFree(d_arr2[i]));
        CUDA_CHECK(cudaFree(res[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
}

int main() {         //! 这里没有修饰符，实际上就是主机函数，这里面可以调用其他的主机函数，也能调用核函数，但是不能调用设备函数
    int n = 100000;  // array size
    int *d_arr;      // 这个变量就是在cpu上
    int *arr2;       // 准备计算一下两个张量之和

    //^ int *arr = new int[n];

    //^ CUDA_CHECK(cudaMalloc(&d_arr, n * sizeof(int)));
    //% 如果这里使用cudaMalloc来进行内存分配，那就会自动分配到GPU上，之后cpu想要访问必须要创建一个在cpu上的变量通过cudaMemcpy进行数据拷贝，将数据从gpu拷贝到cpu上才能访问

    // tip 如果使用cudaMallocManaged来分配内存，那么就可以实现自动统一内存，不需要进行数据拷贝，同样需要cudaFree
    CUDA_CHECK(cudaMallocManaged(&d_arr, n * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&arr2, n * sizeof(int)));

    //@ 常用的block大小（都是 32 的倍数，因为一个 warp 有 32 个线程）
    // int blockSize = 32;   // 小任务
    // int blockSize = 128;  // 中等任务
    const int blockSize = 256;  // 大多数情况推荐
    // int blockSize = 512;  // 大任务
    // int blockSize = 1024; // 最大常用值

    //@ griddim 一般在确定blockdim之后可以推算出来 gridSize = (dataSize + blockSize - 1) / blockSize; 这是向上取整，为了保证数据基本上被完全覆盖，宁愿多一点空闲的空间

    const int gridSize = (n + blockSize - 1) / blockSize;

    std::cout << std::format("gridSize: {}, blockSize: {}\n", gridSize, blockSize);

    auto start = std::chrono::high_resolution_clock::now();
    init_array<<<gridSize, blockSize>>>(d_arr, n);
    init_array<<<gridSize, blockSize>>>(arr2, n);
    CUDA_CHECK(cudaDeviceSynchronize());  //* 这个是用于阻塞主机进程的，因为核函数是异步的，所以需要等待核函数执行完成之后才能稳定获取核函数的结果，其次这个函数还能将错误码进行返回，正常的CUDA代码是不会返回错误，就算代码有问题，程序也能运行，最后得到一个莫名其妙的答案，[由于是异步执行代码，如果想要计算核函数计算时间，就必须要先同步，才能获取到核函数的执行时间，否则你能获得的是核函数的启动时间.]
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);

    std::cout << std::format("Init array time: {} microseconds.\n", duration.count());

    //~ 由于上面的两个数组初始化是允许并行，所以只需要调用一次cudaDeviceSynchronize()即可，如果调用多次，那么就会导致程序过度阻塞，从而影响性能

    start = std::chrono::high_resolution_clock::now();

    add<<<gridSize, blockSize>>>(d_arr, arr2, n);
    CUDA_CHECK(cudaDeviceSynchronize());  // # 这里的加法需要依赖之前的初始化，所以这里的同步不能和前面的同步和在一次，必须分开同步
    duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);

    std::cout << std::format("add array time: {} microseconds.\n", duration.count());

    /*
    start = std::chrono::high_resolution_clock::now();
    e.g cup 对大向量的加法速度
    for(int i = 0; i < n; ++i) {
        arr2[i] += arr2[i];
    }
    duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);

    std::cout << std::format("add array cpu time: {} microseconds.\n", duration.count());

    */

    // 拷贝回主机
    //^ CUDA_CHECK(cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 5; ++i) {
        std::cout << std::format("d_arr:{} <- arr2: {} \n", d_arr[i], arr2[i]);
    }

    std::cout << std::format("=========================== computer sum ========================================\n");

    int *res;
    // give res assigned gpu memory, the res need to sum again to get the sum of the array
    CUDA_CHECK(cudaMallocManaged(&res, gridSize * sizeof(int)));  // the res size is block num, each block has one res
    std::fill(res, res + gridSize, 0);
    start = std::chrono::high_resolution_clock::now();
    int *s;
    CUDA_CHECK(cudaMallocManaged(&s, sizeof(int)));
    *s = 0;
    sum1<<<gridSize, blockSize>>>(arr2, s, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
    std::cout << std::format("sum array time by using globel memory: {} microseconds.  res is {} \n", duration.count(), *s);

    memset(res, 0, gridSize * sizeof(int));
    start = std::chrono::high_resolution_clock::now();
    sum2<<<gridSize, blockSize>>>(arr2, res, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    int sum = 0;
    for (int i = 0; i < gridSize; ++i)
        sum += res[i];
    duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
    std::cout << std::format("sum array time by using share memory: {} microseconds.  res is {} \n", duration.count(), sum);

    memset(res, 0, gridSize * sizeof(int));
    start = std::chrono::high_resolution_clock::now();
    sum3<<<gridSize, blockSize>>>(arr2, res, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    sum = 0;
    for (int i = 0; i < gridSize; ++i)
        sum += res[i];
    duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
    std::cout << std::format("sum array time by using optimized share memory: {} microseconds.  res is {} \n", duration.count(), sum);

    memset(res, 0, gridSize * sizeof(int));
    start = std::chrono::high_resolution_clock::now();
    sum4<blockSize><<<gridSize, blockSize, blockSize * sizeof(int)>>>(arr2, res, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    sum = 0;
    for (int i = 0; i < gridSize; ++i)
        sum += res[i];
    duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
    std::cout << std::format("sum array time by using share memory and warp: {} microseconds.  res is {} \n", duration.count(), sum);

    memset(res, 0, gridSize * sizeof(int));
    start = std::chrono::high_resolution_clock::now();
    sum5<<<gridSize, blockSize>>>(arr2, res, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    sum = 0;
    for (int i = 0; i < gridSize; ++i)
        sum += res[i];
    duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
    std::cout << std::format("sum array time by using warp shuffle: {} microseconds.  res is {} \n", duration.count(), sum);

    std::cout << std::format("real res is {} \n", n * (n - 1) / 2);

    std::cout << std::format("=========================== computer mul ========================================\n");
    int M = 1024, N = 1024, K = 1024;

    // 分配主机内存
    std::vector<float> h_A(M * N, 1.0f);
    std::vector<float> h_B(N * K, 2.0f);
    std::vector<float> h_C(M * K, 0.0f);

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, N * K * sizeof(float));
    cudaMalloc(&d_C, M * K * sizeof(float));

    // 数据传输
    cudaMemcpy(d_A, h_A.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * K * sizeof(float), cudaMemcpyHostToDevice);

    // 配置执行参数
    dim3 threads(16, 16);                          // 16x16 = 256个线程
    dim3 blocks((K + threads.x - 1) / threads.x,   // 还是要注意这里使用的 K 和 M 的顺序, K 和 M 来源于需要覆盖整个网格维度
                (M + threads.y - 1) / threads.y);  // 很明显对于gridDim, 不需要 gridDim.x == gridDim.y
    start = std::chrono::high_resolution_clock::now();
    // 执行核函数
    mul1<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);  // 这里有自动模板类型推导，你不需要显式指定
    CUDA_CHECK(cudaDeviceSynchronize());
    duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
    std::cout << std::format("mul array time by using global memory: {} microseconds. \n", duration.count());

    start = std::chrono::high_resolution_clock::now();
    // 执行核函数
    mul2<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);  // 这里有自动模板类型推导，你不需要显式指定
    CUDA_CHECK(cudaDeviceSynchronize());
    duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
    std::cout << std::format("mul array time by using global memory: {} microseconds. \n", duration.count());

    // 检查结果
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::format("=========================== computer 3D mul ========================================\n");
    int B = 3;

    // 分配主机内存
    // std::vector<float> h_3D_A(B * M * K, 1.0f);
    // std::vector<float> h_3D_B(B * K * N, 2.0f);
    // std::vector<float> h_3D_C(B * M * N, 0.0f);

    // // 分配设备内存
    // cudaMalloc(&d_3D_A, B*M * K * sizeof(float));
    // cudaMalloc(&d_3D_B, B*K * N * sizeof(float));
    // cudaMalloc(&d_3D_C, B*M * N * sizeof(float));

    // // 数据传输
    // cudaMemcpy(d_3D_A, h_3D_A.data(), B*M * K * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_3D_B, h_3D_B.data(), B*K * N * sizeof(float), cudaMemcpyHostToDevice);

    float *d_3D_A, *d_3D_B, *d_3D_C;
    CUDA_CHECK(cudaMallocManaged(&d_3D_A, B * M * N * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&d_3D_B, B * N * K * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&d_3D_C, B * M * K * sizeof(float)));

    std::fill(d_3D_A, d_3D_A + B * M * N, 1.0f);
    std::fill(d_3D_B, d_3D_B + B * N * K, 2.0f);
    std::fill(d_3D_C, d_3D_C + B * M * K, 0.0f);

    // 配置执行参数
    dim3 threads_3D(16, 16);                                                                       // 16x16 = 256个线程
    dim3 blocks_3D((K + threads_3D.x - 1) / threads_3D.x, (M + threads_3D.y - 1) / threads_3D.y);  // 还是要注意这里使用的 N 和 M 的顺序

    start = std::chrono::high_resolution_clock::now();
    // 执行核函数
    mul_3d1<<<blocks_3D, threads_3D>>>(d_3D_A, d_3D_B, d_3D_C, B, M, N, K);  // 这里有自动模板类型推导，你不需要显式指定
    CUDA_CHECK(cudaDeviceSynchronize());
    duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);

    std::cout << std::format("mul 3D array time by using global memory: {} microseconds. \n", duration.count());

    blocks_3D = {(K + threads_3D.x - 1) / threads_3D.x, (M + threads_3D.y - 1) / threads_3D.y, (B + threads_3D.z - 1) / threads_3D.z};  // 还是要注意这里使用的 N 和 M 的顺序

    start = std::chrono::high_resolution_clock::now();
    // 执行核函数
    mul_3d2<<<blocks_3D, threads_3D>>>(d_3D_A, d_3D_B, d_3D_C, B, M, N, K);  // 这里有自动模板类型推导，你不需要显式指定
    CUDA_CHECK(cudaDeviceSynchronize());
    duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);

    std::cout << std::format("mul 3D array time by using share memory: {} microseconds. \n", duration.count());

    for (int i = 0; i < B; ++i) {
        std::cout << "[" << std::endl;
        for (int j = 0; j < 5; ++j) {
            for (int k = 0; k < 5; ++k) {
                std::cout << d_3D_C[i * M * K + j * K + k] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "]" << std::endl;
    }

    std::cout << std::format("=========================== computer stream ========================================\n");
    int martix_num = 50;
    // 分配主机内存
    std::vector<int *> h1(martix_num);
    std::vector<int *> h2(martix_num);
    std::vector<int *> h3(martix_num);
    for (int i = 0; i < martix_num; ++i) {
        h1[i] = new int[B * M * K];
        h2[i] = new int[B * K * N];
        h3[i] = new int[B * M * N];
    }
    const int stream_num = 5;

    auto start_serial = std::chrono::high_resolution_clock::now();
    MulBySerial(h1, h2, h3, B, M, N, K, martix_num);
    auto end_serial = std::chrono::high_resolution_clock::now();

    // 测试流版本
    auto start_streams = std::chrono::high_resolution_clock::now();
    MulByStream(h1, h2, h3, B, M, N, K, martix_num, stream_num);
    auto end_streams = std::chrono::high_resolution_clock::now();

    // 计算耗时
    auto serial_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_serial - start_serial);
    auto streams_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_streams - start_streams);

    // 输出结果
    std::cout << "===== 性能对比 =====" << "\n";
    std::cout << "串行处理时间: " << serial_duration.count() << " ms\n";
    std::cout << "流处理时间 (" << stream_num << " streams): " << streams_duration.count() << " ms\n";
    std::cout << "加速比: " << static_cast<float>(serial_duration.count()) / streams_duration.count() << "x\n";

    // 清理
    for (auto h : h1) delete[] h;
    for (auto h : h2) delete[] h;
    for (auto h : h3) delete[] h;

    cudaFree(d_arr);  // 可以考虑使用智能指针来进行管理
    cudaFree(arr2);
    cudaFree(res);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_3D_A);
    cudaFree(d_3D_B);
    cudaFree(d_3D_C);


    return 0;
}