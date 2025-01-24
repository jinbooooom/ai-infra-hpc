#include "../common/common.h"
#include <stdio.h>
#include <cuda_runtime.h>

/*
 * An example of using CUDA events to control asynchronous work launched on the
 * GPU. In this example, asynchronous copies and an asynchronous kernel are
 * used. A CUDA event is used to determine when that work has completed.
 */

__global__ void kernel(float *g_data, float value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    g_data[idx] = g_data[idx] + value;
}

int checkResult(float *data, const int n, const float x)
{
    for (int i = 0; i < n; i++)
    {
        if (data[i] != x)
        {
            printf("Error! data[%d] = %f, ref = %f\n", i, data[i], x);
            return 0;
        }
    }

    return 1;
}

int main(int argc, char *argv[])
{
    int devID = 0;
    cudaDeviceProp deviceProps;
    CHECK(cudaGetDeviceProperties(&deviceProps, devID));
    printf("> %s running on", argv[0]);
    printf(" CUDA device [%s]\n", deviceProps.name);

    int num = 1 << 24;
    int nbytes = num * sizeof(int);
    float value = 10.0f;

    // allocate host memory
    float *h_a = 0;
    CHECK(cudaMallocHost((void **)&h_a, nbytes));
    memset(h_a, 0, nbytes);

    // allocate device memory
    float *d_a = 0;
    CHECK(cudaMalloc((void **)&d_a, nbytes));
    CHECK(cudaMemset(d_a, 255, nbytes));

    // set kernel launch configuration
    dim3 block = dim3(512);
    dim3 grid  = dim3((num + block.x - 1) / block.x);

    // create cuda event handles
    cudaEvent_t stop;
    CHECK(cudaEventCreate(&stop));

    // asynchronously issue work to the GPU (all to stream 0)
    CHECK(cudaMemcpyAsync(d_a, h_a, nbytes, cudaMemcpyHostToDevice));
    kernel<<<grid, block>>>(d_a, value);
    CHECK(cudaMemcpyAsync(h_a, d_a, nbytes, cudaMemcpyDeviceToHost));
    CHECK(cudaEventRecord(stop));

    // have CPU do some work while waiting for stage 1 to finish
    unsigned long int counter = 0;

    while (cudaEventQuery(stop) == cudaErrorNotReady) {
        counter++;
    }

    // print the cpu and gpu times
    printf("CPU executed %lu iterations while waiting for GPU to finish\n",
           counter);

    // check the output for correctness
    bool bFinalResults = (bool) checkResult(h_a, num, value);

    // release resources
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFreeHost(h_a));
    CHECK(cudaFree(d_a));

    CHECK(cudaDeviceReset());

    exit(bFinalResults ? EXIT_SUCCESS : EXIT_FAILURE);
}
