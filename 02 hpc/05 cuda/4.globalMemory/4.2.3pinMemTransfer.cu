#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/common.h"

/** 
 * 一个使用 CUDA 内存拷贝 API 在设备和主机之间传输数据的示例。
 * 
 * 在这个示例中，使用 cudaMalloc 在 GPU 上分配内存，
 * 并使用 cudaMemcpy 将主机内存中的内容传输到通过 cudaMalloc 分配的数组中。
 * 主机内存通过 cudaMallocHost 分配，以创建一个页锁定（page-locked）的主机数组。
 */

int main(int argc, char **argv)
{
    // set up device
    int dev = getGPUId();
    CHECK(cudaSetDevice(dev));

    // memory size
    unsigned int isize  = 1 << 22;
    unsigned int nbytes = isize * sizeof(float);

    // get device information
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));

    if (!deviceProp.canMapHostMemory)
    {
        printf("Device %d does not support mapping CPU host memory!\n", dev);
        CHECK(cudaDeviceReset());
        exit(EXIT_SUCCESS);
    }

    printf("%s starting at ", argv[0]);
    printf("device %d: %s memory size %d nbyte %5.2fMB, canMap = %d\n", dev, deviceProp.name, isize,
           nbytes / (1024.0f * 1024.0f), deviceProp.canMapHostMemory);

    // allocate pinned host memory
    float *h_a;
    CHECK(cudaMallocHost((float **)&h_a, nbytes));

    // allocate device memory
    float *d_a;
    CHECK(cudaMalloc((float **)&d_a, nbytes));

    // initialize host memory
    memset(h_a, 0, nbytes);

    for (int i = 0; i < isize; i++)
        h_a[i] = 100.10f;

    // transfer data from the host to the device
    CHECK(cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice));

    // transfer data from the device to the host
    CHECK(cudaMemcpy(h_a, d_a, nbytes, cudaMemcpyDeviceToHost));

    // free memory
    CHECK(cudaFree(d_a));
    CHECK(cudaFreeHost(h_a));

    // reset device
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
