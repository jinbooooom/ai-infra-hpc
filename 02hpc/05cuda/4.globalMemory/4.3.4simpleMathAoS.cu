#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * A simple example of using an array of structures to store data on the device.
 * This example is used to study the impact on performance of data layout on the
 * GPU.
 *
 * AoS: one contiguous 64-bit read to get x and y (up to 300 cycles)
 */

#define LEN 1<<22

struct innerStruct
{
    float x;
    float y;
};

struct innerArray
{
    float x[LEN];
    float y[LEN];
};

void initialInnerStruct(innerStruct *ip,  int size)
{
    for (int i = 0; i < size; i++)
    {
        ip[i].x = (float)(rand() & 0xFF) / 100.0f;
        ip[i].y = (float)(rand() & 0xFF) / 100.0f;
    }

    return;
}

void testInnerStructHost(innerStruct *A, innerStruct *C, const int n)
{
    for (int idx = 0; idx < n; idx++)
    {
        C[idx].x = A[idx].x + 10.f;
        C[idx].y = A[idx].y + 20.f;
    }

    return;
}

void checkInnerStruct(innerStruct *hostRef, innerStruct *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i].x - gpuRef[i].x) > epsilon)
        {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i,
                    hostRef[i].x, gpuRef[i].x);
            break;
        }

        if (abs(hostRef[i].y - gpuRef[i].y) > epsilon)
        {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i,
                    hostRef[i].y, gpuRef[i].y);
            break;
        }
    }

    if (!match)  printf("Arrays do not match.\n\n");
}

__global__ void testInnerStruct(innerStruct *data, innerStruct * result,
                                const int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        innerStruct tmp = data[i];
        tmp.x += 10.f;
        tmp.y += 20.f;
        result[i] = tmp;
    }
}

__global__ void warmup(innerStruct *data, innerStruct * result, const int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        innerStruct tmp = data[i];
        tmp.x += 10.f;
        tmp.y += 20.f;
        result[i] = tmp;
    }
}

int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s test struct of array at ", argv[0]);
    printf("device %d: %s \n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // allocate host memory
    int nElem = LEN;
    size_t nBytes = nElem * sizeof(innerStruct);
    innerStruct     *h_A = (innerStruct *)malloc(nBytes);
    innerStruct *hostRef = (innerStruct *)malloc(nBytes);
    innerStruct *gpuRef  = (innerStruct *)malloc(nBytes);

    // initialize host array
    initialInnerStruct(h_A, nElem);
    testInnerStructHost(h_A, hostRef, nElem);

    // allocate device memory
    innerStruct *d_A, *d_C;
    CHECK(cudaMalloc((innerStruct**)&d_A, nBytes));
    CHECK(cudaMalloc((innerStruct**)&d_C, nBytes));

    // copy data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    // set up offset for summaryAU: It is blocksize not offset. Thanks.CZ
    int blocksize = 128;

    if (argc > 1) blocksize = atoi(argv[1]);

    // execution configuration
    dim3 block (blocksize, 1);
    dim3 grid  ((nElem + block.x - 1) / block.x, 1);

    // kernel 1: warmup
    double iStart = seconds();
    warmup<<<grid, block>>>(d_A, d_C, nElem);
    CHECK(cudaDeviceSynchronize());
    double iElaps = seconds() - iStart;
    printf("warmup      <<< %3d, %3d >>> elapsed %f sec\n", grid.x, block.x,
           iElaps);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkInnerStruct(hostRef, gpuRef, nElem);
    CHECK(cudaGetLastError());

    // kernel 2: testInnerStruct
    iStart = seconds();
    testInnerStruct<<<grid, block>>>(d_A, d_C, nElem);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("innerstruct <<< %3d, %3d >>> elapsed %f sec\n", grid.x, block.x,
           iElaps);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkInnerStruct(hostRef, gpuRef, nElem);
    CHECK(cudaGetLastError());

    // free memories both host and device
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_C));
    free(h_A);
    free(hostRef);
    free(gpuRef);

    // reset device
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
