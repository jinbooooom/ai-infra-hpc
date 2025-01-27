#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "../common/common.h"

/*
 * 一个使用 cudaStreamWaitEvent 添加流间依赖的简单示例。
 * 此代码在 n_streams 个流中分别启动 4 个 kernel。
 * 每个流完成时会记录一个事件（kernelEvent）。
 * 然后在该事件和最后一个流（streams[n_streams - 1]）上调用 cudaStreamWaitEvent，
 * 以确保最后一个流中的所有计算仅在所有其他流完成后才执行。
 */

#define N 300000
#define NSTREAM 4

__global__ void kernel_1()
{
    double sum = 0.0;

    for (int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__global__ void kernel_2()
{
    double sum = 0.0;

    for (int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__global__ void kernel_3()
{
    double sum = 0.0;

    for (int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__global__ void kernel_4()
{
    double sum = 0.0;

    for (int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

int main(int argc, char **argv)
{
    int n_streams = NSTREAM;
    int isize     = 1;
    int iblock    = 1;
    int bigcase   = 0;

    // get argument from command line
    if (argc > 1)
        n_streams = atoi(argv[1]);

    if (argc > 2)
        bigcase = atoi(argv[2]);

    float elapsed_time;

    // set up max connectioin
    char *iname = (char *)"CUDA_DEVICE_MAX_CONNECTIONS";
    setenv(iname, "32", 1);
    char *ivalue = getenv(iname);
    printf("%s = %s\n", iname, ivalue);

    int dev = getGPUId();
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("> Using Device %d: %s with num_streams %d\n", dev, deviceProp.name, n_streams);
    CHECK(cudaSetDevice(dev));

    // check if device support hyper-q
    if (deviceProp.major < 3 || (deviceProp.major == 3 && deviceProp.minor < 5))
    {
        if (deviceProp.concurrentKernels == 0)
        {
            printf(
                "> GPU does not support concurrent kernel execution (SM 3.5 "
                "or higher required)\n");
            printf("> CUDA kernel runs will be serialized\n");
        }
        else
        {
            printf("> GPU does not support HyperQ\n");
            printf("> CUDA kernel runs will have limited concurrency\n");
        }
    }

    printf("> Compute Capability %d.%d hardware with %d multi-processors\n", deviceProp.major, deviceProp.minor,
           deviceProp.multiProcessorCount);

    // Allocate and initialize an array of stream handles
    cudaStream_t *streams = (cudaStream_t *)malloc(n_streams * sizeof(cudaStream_t));

    for (int i = 0; i < n_streams; i++)
    {
        CHECK(cudaStreamCreate(&(streams[i])));
    }

    // run kernel with more threads
    if (bigcase == 1)
    {
        iblock = 512;
        isize  = 1 << 12;
    }

    // set up execution configuration
    dim3 block(iblock);
    dim3 grid(isize / iblock);
    printf("> grid %d block %d\n", grid.x, block.x);

    // creat events
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    cudaEvent_t *kernelEvent;
    kernelEvent = (cudaEvent_t *)malloc(n_streams * sizeof(cudaEvent_t));

    for (int i = 0; i < n_streams; i++)
    {
        CHECK(cudaEventCreateWithFlags(&(kernelEvent[i]), cudaEventDisableTiming));
    }

    // record start event
    CHECK(cudaEventRecord(start, 0));

    // dispatch job with depth first ordering
    for (int i = 0; i < n_streams; i++)
    {
        kernel_1<<<grid, block, 0, streams[i]>>>();
        kernel_2<<<grid, block, 0, streams[i]>>>();
        kernel_3<<<grid, block, 0, streams[i]>>>();
        kernel_4<<<grid, block, 0, streams[i]>>>();

        // 第四个流，即streams[n_streams-1]，在其他所有流完成后才能开始启动工作
        CHECK(cudaEventRecord(kernelEvent[i], streams[i]));
        CHECK(cudaStreamWaitEvent(streams[n_streams - 1], kernelEvent[i], 0));
    }

    // record stop event
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));

    // calculate elapsed time
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("Measured time for parallel execution = %.3lfus\n", elapsed_time * 1000.0f);

    // release all stream
    for (int i = 0; i < n_streams; i++)
    {
        CHECK(cudaStreamDestroy(streams[i]));
        CHECK(cudaEventDestroy(kernelEvent[i]));
    }

    free(streams);
    free(kernelEvent);

    // reset device
    CHECK(cudaDeviceReset());

    return 0;
}
