#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/common.h"

/*
 * This example demonstrates the impact of misaligned reads on performance by
 * forcing misaligned reads to occur on a float*. Kernels that reduce the
 * performance impact of misaligned reads via unrolling are also included below.
 */

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match     = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i, hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (!match)
        printf("Arrays do not match.\n\n");
}

void initialData(float *ip, int size)
{
    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 100.0f;
    }

    return;
}

void sumArraysOnHost(float *A, float *B, float *C, const int n, int offset)
{
    for (int idx = offset, k = 0; idx < n; idx++, k++)
    {
        C[k] = A[idx] + B[idx];
    }
}

__global__ void warmup(float *A, float *B, float *C, const int n, int offset)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;

    if (k < n)
        C[i] = A[k] + B[k];
}

__global__ void readOffset(float *A, float *B, float *C, const int n, int offset)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;

    if (k < n)
        C[i] = A[k] + B[k];
}

__global__ void readOffsetUnroll2(float *A, float *B, float *C, const int n, int offset)
{
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    unsigned int k = i + offset;

    if (k < n)
        C[i] = A[k] + B[k];
    if (k + blockDim.x < n)
    {
        C[i + blockDim.x] = A[k + blockDim.x] + B[k + blockDim.x];
    }
}

__global__ void readOffsetUnroll4(float *A, float *B, float *C, const int n, int offset)
{
    unsigned int i = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    unsigned int k = i + offset;

    if (k < n)
        C[i] = A[k] + B[k];
    if (k + blockDim.x < n)
    {
        C[i + blockDim.x] = A[k + blockDim.x] + B[k + blockDim.x];
    }
    if (k + 2 * blockDim.x < n)
    {
        C[i + 2 * blockDim.x] = A[k + 2 * blockDim.x] + B[k + 2 * blockDim.x];
    }
    if (k + 3 * blockDim.x < n)
    {
        C[i + 3 * blockDim.x] = A[k + 3 * blockDim.x] + B[k + 3 * blockDim.x];
    }
}

int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up array size
    int power     = 20;
    int blocksize = 512;
    int offset    = 0;

    if (argc > 1)
        offset = atoi(argv[1]);
    if (argc > 2)
        blocksize = atoi(argv[2]);
    if (argc > 3)
        power = atoi(argv[3]);

    int nElem = 1 << power;  // total number of elements to reduce
    printf(" with array size %d\n", nElem);
    size_t nBytes = nElem * sizeof(float);

    // execution configuration
    dim3 block(blocksize, 1);
    dim3 grid((nElem + block.x - 1) / block.x, 1);

    // allocate host memory
    float *h_A     = (float *)malloc(nBytes);
    float *h_B     = (float *)malloc(nBytes);
    float *hostRef = (float *)malloc(nBytes);
    float *gpuRef  = (float *)malloc(nBytes);

    //  initialize host array
    initialData(h_A, nElem);
    memcpy(h_B, h_A, nBytes);

    //  summary at host side
    sumArraysOnHost(h_A, h_B, hostRef, nElem, offset);

    // allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float **)&d_A, nBytes));
    CHECK(cudaMalloc((float **)&d_B, nBytes));
    CHECK(cudaMalloc((float **)&d_C, nBytes));

    // copy data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_A, nBytes, cudaMemcpyHostToDevice));

    //  kernel 1:
    double iStart = seconds();
    warmup<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    CHECK(cudaDeviceSynchronize());
    double iElaps = seconds() - iStart;
    printf("warmup     <<< %4d, %4d >>> offset %4d elapsed %f sec\n", grid.x, block.x, offset, iElaps);
    CHECK(cudaGetLastError());
    CHECK(cudaMemset(d_C, 0x00, nBytes));

    // kernel 1
    iStart = seconds();
    readOffset<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("readOffset <<< %4d, %4d >>> offset %4d elapsed %f sec\n", grid.x, block.x, offset, iElaps);
    CHECK(cudaGetLastError());

    // copy kernel result back to host side and check device results
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, nElem - offset);
    CHECK(cudaMemset(d_C, 0x00, nBytes));

    // kernel 2
    iStart = seconds();
    readOffsetUnroll2<<<grid.x / 2, block>>>(d_A, d_B, d_C, nElem, offset);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("unroll2    <<< %4d, %4d >>> offset %4d elapsed %f sec\n", grid.x / 2, block.x, offset, iElaps);
    CHECK(cudaGetLastError());

    // copy kernel result back to host side and check device results
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, nElem - offset);
    CHECK(cudaMemset(d_C, 0x00, nBytes));

    // kernel 3
    iStart = seconds();
    readOffsetUnroll4<<<grid.x / 4, block>>>(d_A, d_B, d_C, nElem, offset);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("unroll4    <<< %4d, %4d >>> offset %4d elapsed %f sec\n", grid.x / 4, block.x, offset, iElaps);
    CHECK(cudaGetLastError());

    // copy kernel result back to host side and check device results
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, nElem - offset);
    CHECK(cudaMemset(d_C, 0x00, nBytes));

    // free host and device memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);

    // reset device
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
