#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/common.h"

/*
 * 本示例演示了如何使用 CUDA 事件来控制 GPU 上启动的异步任务。
 * 在此示例中，使用了异步拷贝和异步 kernel。
 * 通过 CUDA 事件来确定这些任务何时完成。
 */

// Kernel 执行向量与标量的加法
__global__ void kernel(float *g_data, float value)
{
    int idx     = blockIdx.x * blockDim.x + threadIdx.x;
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

int doCPUWork()
{
    printf("do CPU work\n");
    return 0;
}

int main(int argc, char *argv[])
{
    int devID = getGPUId();
    cudaDeviceProp deviceProps;
    CHECK(cudaGetDeviceProperties(&deviceProps, devID));
    printf("> %s running on", argv[0]);
    printf(" CUDA device [%s]\n", deviceProps.name);

    int num     = 1 << 24;
    int nbytes  = num * sizeof(int);
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
    // GPU 的核函数与 CPU 的计算工作是并行的。
    // 如果 CPU 先完成，则 CPU 不断询问 GPU kernel 的运行状态。
    // 如果 CPU 耗时更多，则 cudaEventQuery 执行一次就结束了。
    // 总之，重叠了 GPU 与 CPU 的执行。
    doCPUWork();
    unsigned long int counter = 0;

    while (cudaEventQuery(stop) == cudaErrorNotReady)
    {
        counter++;
    }

    // print the cpu and gpu times
    printf("CPU executed %lu iterations while waiting for GPU to finish\n", counter);

    // check the output for correctness
    bool bFinalResults = (bool)checkResult(h_a, num, value);

    // release resources
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFreeHost(h_a));
    CHECK(cudaFree(d_a));

    CHECK(cudaDeviceReset());

    exit(bFinalResults ? EXIT_SUCCESS : EXIT_FAILURE);
}
