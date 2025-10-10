#ifndef CONV_CUH
#define CONV_CUH

#include <cuda_runtime.h>

// 全局内存
__global__ void conv1(float* input, float* kernel, float* output, int input_w, int input_h, int kernel_size, int output_w, int output_h, int stride = 1, int padding = 0) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x >= output_w || out_y >= output_h)
        return;  // 超出输出范围则返回, 本身其他就算是 padding 也会计算出 0

    //@ 计算输入坐标，这个公式有意义，之后在遍历卷积核的过程中是要参考输入坐标点的，所以需要根据输出坐标计算输入坐标
    int input_start_x = out_x * stride - padding;  // 起始
    int input_start_y = out_y * stride - padding;  //

    float sum = 0.0f;  // 寄存器
    int half_kernel_size = kernel_size / 2;
    //~ 遍历卷积核，与对应输入进行加权和
    for (int ky = -half_kernel_size; ky <= half_kernel_size; ++ky) {
        for (int kx = -half_kernel_size; kx <= half_kernel_size; ++kx) {
            int inputy_idx = input_start_y + ky;  // 行 如果有腐蚀参数， inputy_idx = input_start_y + ky * dilation
            int inputx_idx = input_start_x + kx;  // 列
            int kernely_idx = (ky + half_kernel_size);
            int kernelx_idx = (kx + half_kernel_size);
            if (inputy_idx >= 0 && inputy_idx < input_h && inputx_idx >= 0 && inputx_idx < input_w) {
                sum += input[inputy_idx * input_w + inputx_idx] * kernel[kernely_idx * kernel_size + kernelx_idx];
            }
        }
    }
    output[out_y * output_w + out_x] = sum;
}

// 将核存储到常量内存
__constant__ float kernel_data[25];
__global__ void conv2(float* input, float* output, int input_w, int input_h, int kernel_size, int output_w, int output_h, int stride = 1, int padding = 0) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x >= output_w || out_y >= output_h)
        return;  // 超出输出范围则返回, 本身其他就算是 padding 也会计算出 0

    //@ 计算输入坐标，这个公式有意义，之后在遍历卷积核的过程中是要参考输入坐标点的，所以需要根据输出坐标计算输入坐标
    int input_start_x = out_x * stride - padding;
    int input_start_y = out_y * stride - padding;

    float sum = 0.0f;
    int half_kernel_size = kernel_size / 2;
    // 遍历核
    for (int ky = -half_kernel_size; ky <= half_kernel_size; ++ky) {
        for (int kx = -half_kernel_size; kx <= half_kernel_size; ++kx) {
            int input_y_idx = input_start_y + ky;
            int input_x_idx = input_start_x + kx;
            int kernel_idx = (ky + half_kernel_size) * kernel_size + (kx + half_kernel_size);
            if (input_y_idx >= 0 && input_y_idx < input_h && input_x_idx >= 0 && input_x_idx < input_w) {
                sum += input[input_y_idx * input_w + input_x_idx] * kernel_data[kernel_idx];
            }
        }
    }
    output[out_y * output_w + out_x] = sum;
}

// 使用共享内存
__global__ void conv3(float* input, float* output, int input_w, int input_h, int kernel_size, int output_w, int output_h, int stride = 1, int padding = 0) {
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;
    int half_kernel = kernel_size / 2;
    int out_x = blockIdx.x * blockDim.x + local_x;
    int out_y = blockIdx.y * blockDim.y + local_y;

    if (out_x >= output_w || out_y >= output_h) return;
    // 共享内存
    extern __shared__ float sd_input[];

    // 计算当前线程在input上需要计算的索引范围
    int shared_w = blockDim.x * stride + 2 * half_kernel;  // 当前线程需要考虑每次取 shared_w 个行元素 与 卷积进行运算， 这个 2 * half_kernel 是用来填补卷积核中心 与 输入图像边缘 对齐时 溢出的内容（可以称其为“卷积halo”），在shared_data 中将会以 0 进行填充，之所以是这个数值，需要考虑的是一般情况下静态共享内存会开辟出 blockDim.x 这么大，但是考虑到有步长 所以进行了扩充，在外面开辟动态共享内存的时候，也建议开这么大
    int shared_h = blockDim.y * stride + 2 * half_kernel;

    // 还需要和之前的手段一样获取从input的什么地方索引开始获取
    int input_x_start = blockIdx.x * blockDim.x * stride - padding;
    int input_y_start = blockIdx.y * blockDim.y * stride - padding;

    // 将 input 中 x索引 从 input_x_start 到 input_x_start + shared_w 同时 y 索引从 input_y_start 到 input_y_start + shared_h 的数据放到共享内存上, 注意下面的代码将会使用协作加载，不同的线程来拷贝不同的数据，这样能有效防止共享内存的竞争
    for (int i = local_y; i < shared_h; i += blockDim.y) {
        for (int j = local_x; j < shared_w; j += blockDim.x) {
            int input_x_idx = input_x_start + j - half_kernel; // 这里要half_kernel，偏移之后有部分空白就是“卷积halo”
            int input_y_idx = input_y_start + i - half_kernel;
            if(input_x_idx >= 0 && input_x_idx < input_w && input_y_idx >= 0 && input_y_idx < input_h) {
                sd_input[i * shared_w + j] = input[input_y_idx * input_w + input_x_idx];
            }else {
                sd_input[i * shared_w + j] = 0.0f;
            }
        }
    }

    // 同步线程
    __syncthreads();
    // 计算卷积
    float sum = 0;
    for (int ky = -half_kernel; ky <= half_kernel; ++ky) {
        for (int kx = -half_kernel; kx <= half_kernel; ++kx) {
            int kernel_idx = (ky + half_kernel) * kernel_size + (kx + half_kernel);
            // 在共享内存中找元素与其相乘
            int sd_x = local_x * stride + half_kernel + kx;
            int sd_y = local_y * stride + half_kernel + ky;
            sum += sd_input[sd_y * shared_w + sd_x] * kernel_data[kernel_idx];
        }
    }
    // 存入结果
    output[out_y * output_w + out_x] = sum;
}

#endif  // CONV_CUH