#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/common.h"

/*
 * 获取当前 CUDA 平台上第一个设备的基本信息，
 * 包括流多处理器（SM）的数量、常量内存的字节数、
 * 每个块的共享内存字节数等。
 */

int main(int argc, char *argv[])
{
    int iDev = 0;
    cudaDeviceProp iProp;
    CHECK(cudaGetDeviceProperties(&iProp, iDev));

    printf("Device %d: %s\n", iDev, iProp.name);
    printf("  Number of multiprocessors:                     %d\n", iProp.multiProcessorCount);
    printf("  Total amount of constant memory:               %4.2f KB\n", iProp.totalConstMem / 1024.0);
    printf("  Total amount of shared memory per block:       %4.2f KB\n", iProp.sharedMemPerBlock / 1024.0);
    printf("  Total number of registers available per block: %d\n", iProp.regsPerBlock);
    printf("  Warp size:                                     %d\n", iProp.warpSize);
    printf("  Maximum number of threads per block:           %d\n", iProp.maxThreadsPerBlock);
    printf("  Maximum number of threads per multiprocessor:  %d\n", iProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of warps per multiprocessor:    %d\n", iProp.maxThreadsPerMultiProcessor / 32);
    return EXIT_SUCCESS;
}
