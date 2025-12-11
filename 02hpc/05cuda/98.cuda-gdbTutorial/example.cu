#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// 场景1: 简单的向量加法内核
__global__ void vectorAdd(uint32_t *a, uint32_t *b, uint32_t *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 场景2: 带条件分支的内核
__global__ void conditionalKernel(int *data, int n, int threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 这里可以设置断点观察条件分支
        if (data[idx] > threshold) {
            data[idx] = data[idx] * 2;
        } else {
            data[idx] = data[idx] / 2;
        }
    }
}

// 场景3: 使用共享内存的内核
__global__ void sharedMemoryKernel(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 加载数据到共享内存
    if (idx < n) {
        sdata[tid] = input[idx];
    } else {
        sdata[tid] = 0.0f;
    }
    
    __syncthreads();
    
    // 在共享内存上进行计算
    if (idx < n) {
        output[idx] = sdata[tid] * 2.0f;
    }
}

// 场景4: 带循环的内核
__global__ void loopKernel(int *data, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int sum = 0;
        // 循环计算，可以单步调试观察
        for (int i = 0; i < iterations; i++) {
            sum += data[idx] + i;
        }
        data[idx] = sum;
    }
}

// 场景5: 多维线程块的内核
__global__ void matrixKernel(float *matrix, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < height && col < width) {
        int idx = row * width + col;
        matrix[idx] = row * width + col;  // 初始化矩阵
    }
}

// 场景6: 多设备操作（如果有多个GPU）
__global__ void multiDeviceKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 3.14f;
    }
}

// 辅助函数：打印设备信息
void printDeviceInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Found %d CUDA device(s)\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }
}

int main(int argc, char **argv) {
    printf("=== CUDA-GDB Debug Example Program ===\n\n");
    
    // 打印设备信息
    printDeviceInfo();
    
    const int N = 1024;
    const size_t size = N * sizeof(uint32_t);
    
    // 分配主机内存
    uint32_t *h_a = (uint32_t *)malloc(size);
    uint32_t *h_b = (uint32_t *)malloc(size);
    uint32_t *h_c = (uint32_t *)malloc(size);
    
    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_a[i] = i * 1;
        h_b[i] = i * 2;
    }
    
    // 分配设备内存
    uint32_t *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, size));
    CHECK_CUDA(cudaMalloc(&d_b, size));
    CHECK_CUDA(cudaMalloc(&d_c, size));
    
    // 复制数据到设备
    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));
    
    printf("\n--- Scenario 1: Vector Addition ---\n");
    // 场景1: 向量加法
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 复制结果回主机
    CHECK_CUDA(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));
    printf("First 5 results: %u, %u, %u, %u, %u\n", 
           h_c[0], h_c[1], h_c[2], h_c[3], h_c[4]);
    
    printf("\n--- Scenario 2: Conditional Branch ---\n");
    // 场景2: 条件分支
    int *d_data;
    int *h_data = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        h_data[i] = i;
    }
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice));
    
    int threshold = 500;
    conditionalKernel<<<gridSize, blockSize>>>(d_data, N, threshold);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost));
    printf("Threshold=%d, First 5 results: %d, %d, %d, %d, %d\n", 
           threshold, h_data[0], h_data[1], h_data[2], h_data[3], h_data[4]);
    
    printf("\n--- Scenario 3: Shared Memory ---\n");
    // 场景3: 共享内存
    float *d_input, *d_output;
    size_t floatSize = N * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_input, floatSize));
    CHECK_CUDA(cudaMalloc(&d_output, floatSize));
    
    float *h_input = (float *)malloc(floatSize);
    float *h_output = (float *)malloc(floatSize);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)h_a[i];
    }
    CHECK_CUDA(cudaMemcpy(d_input, h_input, floatSize, cudaMemcpyHostToDevice));
    
    size_t sharedMemSize = blockSize.x * sizeof(float);
    sharedMemoryKernel<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaMemcpy(h_output, d_output, floatSize, cudaMemcpyDeviceToHost));
    printf("First 5 results after shared memory computation: %.2f, %.2f, %.2f, %.2f, %.2f\n", 
           h_output[0], h_output[1], h_output[2], h_output[3], h_output[4]);
    free(h_input);
    free(h_output);
    
    printf("\n--- Scenario 4: Loop Computation ---\n");
    // 场景4: 循环
    int iterations = 10;
    loopKernel<<<gridSize, blockSize>>>(d_data, N, iterations);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost));
    printf("First 5 results after %d iterations: %d, %d, %d, %d, %d\n", 
           iterations, h_data[0], h_data[1], h_data[2], h_data[3], h_data[4]);
    
    printf("\n--- Scenario 5: Multi-dimensional Thread Blocks ---\n");
    // 场景5: 矩阵操作
    const int width = 32;
    const int height = 32;
    float *d_matrix;
    CHECK_CUDA(cudaMalloc(&d_matrix, width * height * sizeof(float)));
    
    dim3 blockSize2D(8, 8);
    dim3 gridSize2D((width + blockSize2D.x - 1) / blockSize2D.x,
                    (height + blockSize2D.y - 1) / blockSize2D.y);
    matrixKernel<<<gridSize2D, blockSize2D>>>(d_matrix, width, height);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    float *h_matrix = (float *)malloc(width * height * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_matrix, d_matrix, width * height * sizeof(float), 
                          cudaMemcpyDeviceToHost));
    printf("Matrix[0][0]=%.0f, Matrix[0][1]=%.0f, Matrix[1][0]=%.0f\n", 
           h_matrix[0], h_matrix[1], h_matrix[width]);
    
    printf("\n--- Scenario 6: Multi-device Operation ---\n");
    // 场景6: 多设备（如果有）
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 1) {
        printf("Multiple devices detected, switching to device 1\n");
        cudaSetDevice(1);
        float *d_data2;
        size_t floatSize2 = N * sizeof(float);
        CHECK_CUDA(cudaMalloc(&d_data2, floatSize2));
        float *h_data2 = (float *)malloc(floatSize2);
        for (int i = 0; i < N; i++) {
            h_data2[i] = (float)h_a[i];
        }
        CHECK_CUDA(cudaMemcpy(d_data2, h_data2, floatSize2, cudaMemcpyHostToDevice));
        multiDeviceKernel<<<gridSize, blockSize>>>(d_data2, N);
        CHECK_CUDA(cudaDeviceSynchronize());
        free(h_data2);
        cudaFree(d_data2);
        cudaSetDevice(0);
    } else {
        printf("Only one device found, skipping multi-device test\n");
    }
    
    // 清理
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_data);
    free(h_matrix);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_data);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_matrix);
    
    printf("\n=== Program execution completed ===\n");
    return 0;
}




