#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <thread>
#include <mutex>

#define TEST_COUNT 10

// NVTX 辅助函数：创建带颜色的 CPU 范围标记
void nvtxMarkCPURange(const char* name) {
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = 0xFF00FF00;  // 绿色 (ARGB格式: 0xAARRGGBB)
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = name;
    nvtxRangePushEx(&eventAttrib);
}

// 核函数1: 向量加法
__global__ void vectorAdd(float *A, float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// 核函数2: 向量乘法
__global__ void vectorMultiply(float *A, float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] * B[idx];
    }
}

// 核函数3: 向量点积（归约操作，较复杂）
__global__ void vectorDotProduct(float *A, float *B, float *result, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 每个线程计算一个元素
    sdata[tid] = (idx < N) ? A[idx] * B[idx] : 0.0f;
    __syncthreads();
    
    // 归约求和
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 第一个线程写入结果
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

// 核函数4: 简单的数学运算（正弦和余弦）
__global__ void vectorSinCos(float *A, float *B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = A[idx];
        A[idx] = sinf(val);
        B[idx] = cosf(val);
    }
}

// 核函数5: 向量缩放
__global__ void vectorScale(float *A, float scale, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        A[idx] = A[idx] * scale;
    }
}

// ========== CPU 版本的函数（串行计算） ==========

// CPU函数1: 向量加法
void cpuVectorAdd(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

// CPU函数2: 向量乘法
void cpuVectorMultiply(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] * B[i];
    }
}

// CPU函数3: 向量点积
void cpuVectorDotProduct(float *A, float *B, float *result, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += A[i] * B[i];
    }
    *result = sum;
}

// CPU函数4: 正弦余弦计算
void cpuVectorSinCos(float *A, float *B, int N) {
    for (int i = 0; i < N; i++) {
        float val = A[i];
        A[i] = sinf(val);
        B[i] = cosf(val);
    }
}

// CPU函数5: 向量缩放
void cpuVectorScale(float *A, float scale, int N) {
    for (int i = 0; i < N; i++) {
        A[i] = A[i] * scale;
    }
}

// 检查CUDA错误
void checkCudaError(cudaError_t error, const char *msg) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error %s: %s\n", msg, cudaGetErrorString(error));
        exit(1);
    }
}

// CPU计算线程函数
void cpuComputeThread(float *h_A_cpu, float *h_B_cpu, float *h_C_cpu, 
                      float *h_result_cpu, int N, float scale) {
    for (int i = 0; i < TEST_COUNT; i++) {
        // CPU函数1: 向量加法
        nvtxMarkCPURange("CPU: vectorAdd");
        cpuVectorAdd(h_A_cpu, h_B_cpu, h_C_cpu, N);
        nvtxRangePop();
        
        // CPU函数2: 向量乘法
        nvtxMarkCPURange("CPU: vectorMultiply");
        cpuVectorMultiply(h_A_cpu, h_B_cpu, h_C_cpu, N);
        nvtxRangePop();
        
        // CPU函数3: 向量点积
        nvtxMarkCPURange("CPU: vectorDotProduct");
        cpuVectorDotProduct(h_A_cpu, h_B_cpu, h_result_cpu, N);
        nvtxRangePop();
        
        // CPU函数4: 正弦余弦计算
        nvtxMarkCPURange("CPU: vectorSinCos");
        cpuVectorSinCos(h_A_cpu, h_B_cpu, N);
        nvtxRangePop();
        
        // CPU函数5: 向量缩放
        nvtxMarkCPURange("CPU: vectorScale");
        cpuVectorScale(h_A_cpu, scale, N);
        nvtxRangePop();
    }
}

int main(int argc, char *argv[]) {
    // 设置数组大小
    int N = 1024 * 1024;  // 1M 元素
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    
    size_t size = N * sizeof(float);
    
    printf("Nsight Tutorial Program\n");
    printf("Array size: %d elements (%zu bytes)\n", N, size);
    
    // 查询并显示 GPU 信息
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "Error: No CUDA device detected\n");
        return 1;
    }
    
    int deviceId = 0;  // 默认使用设备 0
    // 可以通过命令行参数指定设备，例如: ./nsight_tutorial 1048576 1
    if (argc > 2) {
        deviceId = atoi(argv[2]);
        if (deviceId < 0 || deviceId >= deviceCount) {
            fprintf(stderr, "Warning: Invalid device ID %d, using default device 0\n", deviceId);
            deviceId = 0;
        }
    }
    
    // 显式设置使用的 GPU 设备
    checkCudaError(cudaSetDevice(deviceId), "Set CUDA device");
    
    // 获取并显示当前设备信息
    cudaDeviceProp deviceProp;
    checkCudaError(cudaGetDeviceProperties(&deviceProp, deviceId), "Get device properties");
    printf("Using GPU device: %d / %d\n", deviceId, deviceCount - 1);
    printf("Device name: %s\n", deviceProp.name);
    printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Global memory: %.2f GB\n", deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Number of multiprocessors: %d\n\n", deviceProp.multiProcessorCount);
    
    // 分配主机内存
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    float *h_result = (float *)malloc(sizeof(float));
    
    // 初始化数据
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }
    *h_result = 0.0f;
    
    // 分配设备内存
    float *d_A, *d_B, *d_C, *d_result;
    checkCudaError(cudaMalloc(&d_A, size), "Allocate d_A");
    checkCudaError(cudaMalloc(&d_B, size), "Allocate d_B");
    checkCudaError(cudaMalloc(&d_C, size), "Allocate d_C");
    checkCudaError(cudaMalloc(&d_result, sizeof(float)), "Allocate d_result");
    
    // 复制数据到设备
    checkCudaError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), "Copy A to device");
    checkCudaError(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice), "Copy B to device");
    
    // 设置线程块大小
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // CPU计算的部分
    // 为 CPU 计算准备独立的数据缓冲区（避免与 GPU 数据冲突）
    float *h_A_cpu = (float *)malloc(size);
    float *h_B_cpu = (float *)malloc(size);
    float *h_C_cpu = (float *)malloc(size);
    float *h_result_cpu = (float *)malloc(sizeof(float));
    
    // 复制数据到 CPU 缓冲区
    for (int i = 0; i < N; i++) {
        h_A_cpu[i] = h_A[i];
        h_B_cpu[i] = h_B[i];
    }
    *h_result_cpu = 0.0f;
    
    float scale = 2.5f;
    
    // 在单独的线程中启动 CPU 计算（在 GPU 核函数之前）
    std::thread cpuThread(cpuComputeThread, h_A_cpu, h_B_cpu, h_C_cpu, 
                          h_result_cpu, N, scale);
    
    // GPU 核函数启动（与 CPU 线程并行执行）
    for (int i = 0; i < TEST_COUNT; i++) {
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

        vectorMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        
        // 核函数3: 向量点积（需要共享内存）
        size_t sharedMemSize = threadsPerBlock * sizeof(float);
        vectorDotProduct<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_A, d_B, d_result, N);
    
        vectorSinCos<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N);
        
        vectorScale<<<blocksPerGrid, threadsPerBlock>>>(d_A, scale, N);
    }
    
    // 等待 CPU 计算线程完成
    printf("Waiting for CPU computation thread to complete...\n");
    cpuThread.join();
    
    // 清理 CPU 缓冲区
    free(h_A_cpu);
    free(h_B_cpu);
    free(h_C_cpu);
    free(h_result_cpu);
    
    // 同步所有操作
    cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(), "Final synchronization");
    
    printf("Execution completed!\n");
    printf("\nTip: Use the following commands to analyze performance:\n");
    printf("  nsys profile --trace=cuda,nvtx,osrt ./nsightTutorial\n");
    printf("  or\n");
    printf("  ncu --set full ./nsightTutorial\n");
    printf("\nTip: You can specify device via command line arguments:\n");
    printf("  ./nsightTutorial <array_size> <device_id>\n");
    printf("  Example: ./nsightTutorial 1048576 0  (use device 0)\n");
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_result);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_result);
    
    return 0;
}

