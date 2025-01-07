#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

// 定义 CUDA 核函数，修饰符__global__告诉编译器这个函数将会从CPU中调用，然后在GPU上执行
__global__ void helloCUDA()
{
    printf(
        "threadIdx:(%d, %d, %d), blockIdx:(%d, %d, %d), blockDim:(%d, %d, %d), gridDim:(%d, %d, %d), Hello, World!\n",
        threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z,
        gridDim.x, gridDim.y, gridDim.z);
}

int main()
{
    /*
    调用 CUDA 核函数三重尖括号意味着从主线程到设备端代码的调用。
    一个内核函数通过一组线程来执行，所有线程执行相同的代码。
    三重尖括号里面的参数是执行配置，用来说明 Kernel 使用的线程以及结构。
    在这个例子中，有 1 * 2 * 2 * 3 = 12 个 GPU 线程被调用。
    */
    dim3 grid(1, 2);
    dim3 block(2, 3);
    helloCUDA<<<grid, block>>>();

    // 等待 GPU 执行完毕
    cudaDeviceSynchronize();

    return 0;
}

/*
输出：
threadIdx:(0, 0, 0), blockIdx:(0, 0, 0), blockDim:(2, 3, 1), gridDim:(1, 2, 1), Hello, World!
threadIdx:(1, 0, 0), blockIdx:(0, 0, 0), blockDim:(2, 3, 1), gridDim:(1, 2, 1), Hello, World!
threadIdx:(0, 1, 0), blockIdx:(0, 0, 0), blockDim:(2, 3, 1), gridDim:(1, 2, 1), Hello, World!
threadIdx:(1, 1, 0), blockIdx:(0, 0, 0), blockDim:(2, 3, 1), gridDim:(1, 2, 1), Hello, World!
threadIdx:(0, 2, 0), blockIdx:(0, 0, 0), blockDim:(2, 3, 1), gridDim:(1, 2, 1), Hello, World!
threadIdx:(1, 2, 0), blockIdx:(0, 0, 0), blockDim:(2, 3, 1), gridDim:(1, 2, 1), Hello, World!
threadIdx:(0, 0, 0), blockIdx:(0, 1, 0), blockDim:(2, 3, 1), gridDim:(1, 2, 1), Hello, World!
threadIdx:(1, 0, 0), blockIdx:(0, 1, 0), blockDim:(2, 3, 1), gridDim:(1, 2, 1), Hello, World!
threadIdx:(0, 1, 0), blockIdx:(0, 1, 0), blockDim:(2, 3, 1), gridDim:(1, 2, 1), Hello, World!
threadIdx:(1, 1, 0), blockIdx:(0, 1, 0), blockDim:(2, 3, 1), gridDim:(1, 2, 1), Hello, World!
threadIdx:(0, 2, 0), blockIdx:(0, 1, 0), blockDim:(2, 3, 1), gridDim:(1, 2, 1), Hello, World!
threadIdx:(1, 2, 0), blockIdx:(0, 1, 0), blockDim:(2, 3, 1), gridDim:(1, 2, 1), Hello, World!
*/
