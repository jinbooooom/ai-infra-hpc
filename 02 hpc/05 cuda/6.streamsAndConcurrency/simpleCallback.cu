#include "../common/common.h"
#include <stdio.h>
#include <cuda_runtime.h>

/*
 * An example of using CUDA callbacks to trigger work on the host after the
 * completion of asynchronous work on the device. In this example, n_streams
 * CUDA streams are created and 4 kernels are launched asynchronously in each.
 * Then, a callback is added at the completion of those asynchronous kernels
 * that prints diagnostic information.
 */

#define N 100000
#define NSTREAM 4

void CUDART_CB my_callback(cudaStream_t stream, cudaError_t status, void *data)
{
    printf("callback from stream %d\n", *((int *)data));
}

__global__ void kernel_1()
{
    double sum = 0.0;

    for(int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__global__ void kernel_2()
{
    double sum = 0.0;

    for(int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__global__ void kernel_3()
{
    double sum = 0.0;

    for(int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__global__ void kernel_4()
{
    double sum = 0.0;

    for(int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

int main(int argc, char **argv)
{
    int n_streams = NSTREAM;

    if (argc > 1) n_streams = atoi(argv[1]);

    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("> %s Starting...\n", argv[0]);
    printf("> Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // check if device support hyper-q
    if (deviceProp.major < 3 || (deviceProp.major == 3 && deviceProp.minor < 5))
    {
        if (deviceProp.concurrentKernels == 0)
        {
            printf("> GPU does not support concurrent kernel execution (SM 3.5 "
                   "or higher required)\n");
            printf("> CUDA kernel runs will be serialized\n");
        }
        else
        {
            printf("> GPU does not support HyperQ\n");
            printf("> CUDA kernel runs will have limited concurrency\n");
        }
    }

    printf("> Compute Capability %d.%d hardware with %d multi-processors\n",
           deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

    // set up max connectioin
    char * iname = "CUDA_DEVICE_MAX_CONNECTIONS";
    setenv (iname, "8", 1);
    char *ivalue =  getenv (iname);
    printf ("> %s = %s\n", iname, ivalue);
    printf ("> with streams = %d\n", n_streams);

    // Allocate and initialize an array of stream handles
    cudaStream_t *streams = (cudaStream_t *) malloc(n_streams * sizeof(
                                cudaStream_t));

    for (int i = 0 ; i < n_streams ; i++)
    {
        CHECK(cudaStreamCreate(&(streams[i])));
    }

    dim3 block (1);
    dim3 grid  (1);
    cudaEvent_t start_event, stop_event;
    CHECK(cudaEventCreate(&start_event));
    CHECK(cudaEventCreate(&stop_event));

    int stream_ids[n_streams];

    CHECK(cudaEventRecord(start_event, 0));

    for (int i = 0; i < n_streams; i++)
    {
        stream_ids[i] = i;
        kernel_1<<<grid, block, 0, streams[i]>>>();
        kernel_2<<<grid, block, 0, streams[i]>>>();
        kernel_3<<<grid, block, 0, streams[i]>>>();
        kernel_4<<<grid, block, 0, streams[i]>>>();
        CHECK(cudaStreamAddCallback(streams[i], my_callback,
                    (void *)(stream_ids + i), 0));
    }

    CHECK(cudaEventRecord(stop_event, 0));
    CHECK(cudaEventSynchronize(stop_event));

    float elapsed_time;
    CHECK(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
    printf("Measured time for parallel execution = %.3fs\n",
           elapsed_time / 1000.0f);

    // release all stream
    for (int i = 0 ; i < n_streams ; i++)
    {
        CHECK(cudaStreamDestroy(streams[i]));
    }

    free(streams);

    /*
     * cudaDeviceReset must be called before exiting in order for profiling and
     * tracing tools such as Nsight and Visual Profiler to show complete traces.
     */
    CHECK(cudaDeviceReset());

    return 0;
}
