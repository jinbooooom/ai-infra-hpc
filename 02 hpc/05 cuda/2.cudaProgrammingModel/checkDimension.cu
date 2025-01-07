#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/common.h"

/*
 * 显示线程块与线程网格
 */

__global__ void checkIndex(void)
{
    printf("threadIdx:(%d, %d, %d), blockIdx:(%d, %d, %d), blockDim:(%d, %d, %d), gridDim:(%d, %d, %d)\n", threadIdx.x,
           threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z, gridDim.x,
           gridDim.y, gridDim.z);
}

int main(int argc, char **argv)
{
    // define total data element
    int nElem = 12;

    // define grid and block structure
    dim3 block(3);
    dim3 grid((nElem + block.x - 1) / block.x);

    // check grid and block dimension from host side
    printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
    printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);

    // check grid and block dimension from device side
    checkIndex<<<grid, block>>>();

    // reset device before you leave
    // 用于重置当前设备（GPU）。它的主要作用是清理与该设备相关的所有资源，并将设备恢复到初始状态
    CHECK(cudaDeviceReset());

    return (0);
}

/*
grid.x 4 grid.y 1 grid.z 1
block.x 3 block.y 1 block.z 1
threadIdx:(0, 0, 0), blockIdx:(0, 0, 0), blockDim:(3, 1, 1), gridDim:(4, 1, 1)
threadIdx:(1, 0, 0), blockIdx:(0, 0, 0), blockDim:(3, 1, 1), gridDim:(4, 1, 1)
threadIdx:(2, 0, 0), blockIdx:(0, 0, 0), blockDim:(3, 1, 1), gridDim:(4, 1, 1)
threadIdx:(0, 0, 0), blockIdx:(1, 0, 0), blockDim:(3, 1, 1), gridDim:(4, 1, 1)
threadIdx:(1, 0, 0), blockIdx:(1, 0, 0), blockDim:(3, 1, 1), gridDim:(4, 1, 1)
threadIdx:(2, 0, 0), blockIdx:(1, 0, 0), blockDim:(3, 1, 1), gridDim:(4, 1, 1)
threadIdx:(0, 0, 0), blockIdx:(3, 0, 0), blockDim:(3, 1, 1), gridDim:(4, 1, 1)
threadIdx:(1, 0, 0), blockIdx:(3, 0, 0), blockDim:(3, 1, 1), gridDim:(4, 1, 1)
threadIdx:(2, 0, 0), blockIdx:(3, 0, 0), blockDim:(3, 1, 1), gridDim:(4, 1, 1)
threadIdx:(0, 0, 0), blockIdx:(2, 0, 0), blockDim:(3, 1, 1), gridDim:(4, 1, 1)
threadIdx:(1, 0, 0), blockIdx:(2, 0, 0), blockDim:(3, 1, 1), gridDim:(4, 1, 1)
threadIdx:(2, 0, 0), blockIdx:(2, 0, 0), blockDim:(3, 1, 1), gridDim:(4, 1, 1)
*/