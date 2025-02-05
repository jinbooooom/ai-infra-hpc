#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/common.h"

/*
 * 一个简单的 GPU 嵌套内核启动示例。每个线程在执行开始时显示其信息，
 * 并在下一层嵌套完成时显示诊断信息。
 */

// iDepth 指递归的深度
__global__ void nestedHelloWorld(int const iSize, int iDepth)
{
    int tid = threadIdx.x;
    printf("Recursion=%d: Hello World from thread %d block %d\n", iDepth, tid, blockIdx.x);

    // condition to stop recursive execution
    if (iSize == 1)
        return;

    // reduce block size to half
    int nthreads = iSize >> 1;

    // thread 0 launches child grid recursively
    if (tid == 0 && nthreads > 0)
    {
        nestedHelloWorld<<<1, nthreads>>>(nthreads, ++iDepth);
        printf("-------> nested execution depth: %d\n", iDepth);
    }
}

int main(int argc, char **argv)
{
    int size      = 8;
    int blocksize = 8;  // initial block size
    int igrid     = 1;

    if (argc > 1)
    {
        igrid = atoi(argv[1]);
        size  = igrid * blocksize;
    }

    dim3 block(blocksize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    printf("%s Execution Configuration: grid %d block %d\n", argv[0], grid.x, block.x);

    nestedHelloWorld<<<grid, block>>>(block.x, 0);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceReset());
    return 0;
}
