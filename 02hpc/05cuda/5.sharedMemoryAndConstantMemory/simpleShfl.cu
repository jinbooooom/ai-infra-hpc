#include "../common/common.h"
#include <stdio.h>
#include <cuda_runtime.h>

/*
 * This file includes simple demonstrations of a variety of shuffle
 * instructions.
 */

#define BDIMX 16
#define SEGM  4

void printData(int *in, const int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%2d ", in[i]);
    }

    printf("\n");
}

__global__ void test_shfl_broadcast(int *d_out, int *d_in, int const srcLane)
{
    int value = d_in[threadIdx.x];
    value = __shfl(value, srcLane, BDIMX);
    d_out[threadIdx.x] = value;
}

__global__ void test_shfl_wrap (int *d_out, int *d_in, int const offset)
{
    int value = d_in[threadIdx.x];
    value = __shfl(value, threadIdx.x + offset, BDIMX);
    d_out[threadIdx.x] = value;
}

__global__ void test_shfl_wrap_plus (int *d_out, int *d_in, int const offset)
{
    int value = d_in[threadIdx.x];
    value += __shfl(value, threadIdx.x + offset, BDIMX);
    d_out[threadIdx.x] = value;
}


__global__ void test_shfl_up(int *d_out, int *d_in, unsigned int const delta)
{
    int value = d_in[threadIdx.x];
    value = __shfl_up (value, delta, BDIMX);
    d_out[threadIdx.x] = value;
}

__global__ void test_shfl_down(int *d_out, int *d_in, unsigned int const delta)
{
    int value = d_in[threadIdx.x];
    value = __shfl_down (value, delta, BDIMX);
    d_out[threadIdx.x] = value;
}

__global__ void test_shfl_xor(int *d_out, int *d_in, int const mask)
{
    int value = d_in[threadIdx.x];
    value = __shfl_xor (value, mask, BDIMX);
    d_out[threadIdx.x] = value;
}

__global__ void test_shfl_xor_array(int *d_out, int *d_in, int const mask)
{
    int idx = threadIdx.x * SEGM;
    int value[SEGM];

    for (int i = 0; i < SEGM; i++) value[i] = d_in[idx + i];

    value[0] = __shfl_xor (value[0], mask, BDIMX);
    value[1] = __shfl_xor (value[1], mask, BDIMX);
    value[2] = __shfl_xor (value[2], mask, BDIMX);
    value[3] = __shfl_xor (value[3], mask, BDIMX);

    for (int i = 0; i < SEGM; i++) d_out[idx + i] = value[i];
}


__global__ void test_shfl_xor_int4(int *d_out, int *d_in, int const mask)
{
    int idx = threadIdx.x * SEGM;
    int4 value;

    value.x = d_in[idx];
    value.y = d_in[idx + 1];
    value.z = d_in[idx + 2];
    value.w = d_in[idx + 3];

    value.x = __shfl_xor (value.x, mask, BDIMX);
    value.y = __shfl_xor (value.y, mask, BDIMX);
    value.z = __shfl_xor (value.z, mask, BDIMX);
    value.w = __shfl_xor (value.w, mask, BDIMX);

    d_out[idx] = value.x;
    d_out[idx + 1] = value.y;
    d_out[idx + 2] = value.z;
    d_out[idx + 3] = value.w;
}



__global__ void test_shfl_xor_element(int *d_out, int *d_in, int const mask,
                                      int srcIdx, int dstIdx)
{
    int idx = threadIdx.x * SEGM;
    int value[SEGM];

    for (int i = 0; i < SEGM; i++) value[i] = d_in[idx + i];

    value[srcIdx] = __shfl_xor (value[dstIdx], mask, BDIMX);

    for (int i = 0; i < SEGM; i++) d_out[idx + i] = value[i];
}


__global__ void test_shfl_xor_array_swap (int *d_out, int *d_in, int const mask,
        int srcIdx, int dstIdx)
{
    int idx = threadIdx.x * SEGM;
    int value[SEGM];

    for (int i = 0; i < SEGM; i++) value[i] = d_in[idx + i];

    bool pred = ((threadIdx.x & 1) != mask);

    if (pred)
    {
        int tmp = value[srcIdx];
        value[srcIdx] = value[dstIdx];
        value[dstIdx] = tmp;
    }

    value[dstIdx] = __shfl_xor (value[dstIdx], mask, BDIMX);

    if (pred)
    {
        int tmp = value[srcIdx];
        value[srcIdx] = value[dstIdx];
        value[dstIdx] = tmp;
    }

    for (int i = 0; i < SEGM; i++) d_out[idx + i] = value[i];
}


__inline__ __device__
void swap_old(int *value, int tid, int mask, int srcIdx, int dstIdx)
{
    bool pred = ((tid / mask + 1) == 1);

    if (pred)
    {
        int tmp = value[srcIdx];
        value[srcIdx] = value[dstIdx];
        value[dstIdx] = tmp;
    }

    value[dstIdx] = __shfl_xor (value[dstIdx], mask, BDIMX);

    if (pred)
    {
        int tmp = value[srcIdx];
        value[srcIdx] = value[dstIdx];
        value[dstIdx] = tmp;
    }
}

__inline__ __device__
void swap(int *value, int laneIdx, int mask, int firstIdx, int secondIdx)
{
    bool pred = ((laneIdx / mask + 1) == 1);

    if (pred)
    {
        int tmp = value[firstIdx];
        value[firstIdx] = value[secondIdx];
        value[secondIdx] = tmp;
    }

    value[secondIdx] = __shfl_xor (value[secondIdx], mask, BDIMX);

    if (pred)
    {
        int tmp = value[firstIdx];
        value[firstIdx] = value[secondIdx];
        value[secondIdx] = tmp;
    }
}

__global__ void test_shfl_swap_old (int *d_out, int *d_in, int const mask,
                                    int srcIdx, int dstIdx)
{
    int idx = threadIdx.x * SEGM;
    int value[SEGM];

    for (int i = 0; i < SEGM; i++) value[i] = d_in[idx + i];

    swap(value, threadIdx.x, mask, srcIdx, dstIdx);

    for (int i = 0; i < SEGM; i++) d_out[idx + i] = value[i];
}

__global__
void test_shfl_swap (int *d_out, int *d_in, int const mask, int firstIdx,
                     int secondIdx)
{
    int idx = threadIdx.x * SEGM;
    int value[SEGM];

    for (int i = 0; i < SEGM; i++) value[i] = d_in[idx + i];

    swap(value, threadIdx.x, mask, firstIdx, secondIdx);

    for (int i = 0; i < SEGM; i++) d_out[idx + i] = value[i];
}


__global__ void test_shfl_xor_array_swap_base (int *d_out, int *d_in,
        int const mask, int srcIdx, int dstIdx)
{
    int idx = threadIdx.x * SEGM;
    int value[SEGM];

    for (int i = 0; i < SEGM; i++) value[i] = d_in[idx + i];

    value[dstIdx] = __shfl_xor (value[dstIdx], mask, BDIMX);

    for (int i = 0; i < SEGM; i++) d_out[idx + i] = value[i];
}

__global__ void test_shfl_array(int *d_out, int *d_in, int const offset)
{
    int idx = threadIdx.x * SEGM;
    int value[SEGM];

    for (int i = 0; i < SEGM; i++) value[i] = d_in[idx + i];

    int lane =  (offset + threadIdx.x) % SEGM;
    value[0] = __shfl (value[3], lane, BDIMX);

    for (int i = 0; i < SEGM; i++) d_out[idx + i] = value[i];
}

__global__ void test_shfl_xor_plus(int *d_out, int *d_in, int const mask)
{
    int value = d_in[threadIdx.x];
    value += __shfl_xor (value, mask, BDIMX);
    d_out[threadIdx.x] = value;
}

int main(int argc, char **argv)
{
    int dev = 0;
    bool iPrintout = 1;

    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("> %s Starting.", argv[0]);
    printf("at Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int nElem = BDIMX;
    int h_inData[BDIMX], h_outData[BDIMX];

    for (int i = 0; i < nElem; i++) h_inData[i] = i;

    if(iPrintout)
    {
        printf("initialData\t\t: ");
        printData(h_inData, nElem);
    }

    size_t nBytes = nElem * sizeof(int);
    int *d_inData, *d_outData;
    CHECK(cudaMalloc((int**)&d_inData, nBytes));
    CHECK(cudaMalloc((int**)&d_outData, nBytes));

    CHECK(cudaMemcpy(d_inData, h_inData, nBytes, cudaMemcpyHostToDevice));

    int block = BDIMX;

    // shfl bcast
    test_shfl_broadcast<<<1, block>>>(d_outData, d_inData, 2);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if(iPrintout)
    {
        printf("shfl bcast\t\t: ");
        printData(h_outData, nElem);
    }

    // shfl offset
    test_shfl_wrap<<<1, block>>>(d_outData, d_inData, -2);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if(iPrintout)
    {
        printf("shfl wrap right\t\t: ");
        printData(h_outData, nElem);
    }

    // shfl up
    test_shfl_up<<<1, block>>>(d_outData, d_inData, 2);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if(iPrintout)
    {
        printf("shfl up \t\t: ");
        printData(h_outData, nElem);
    }

    // shfl offset
    test_shfl_wrap<<<1, block>>>(d_outData, d_inData, 2);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if(iPrintout)
    {
        printf("shfl wrap left\t\t: ");
        printData(h_outData, nElem);
    }

    // shfl offset
    test_shfl_wrap<<<1, block>>>(d_outData, d_inData, 2);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if(iPrintout)
    {
        printf("shfl wrap 2\t\t: ");
        printData(h_outData, nElem);
    }

    // shfl down
    test_shfl_down<<<1, block>>>(d_outData, d_inData, 2);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if(iPrintout)
    {
        printf("shfl down \t\t: ");
        printData(h_outData, nElem);
    }

    // shfl xor
    test_shfl_xor<<<1, block>>>(d_outData, d_inData, 1);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if(iPrintout)
    {
        printf("initialData\t\t: ");
        printData(h_inData, nElem);
    }

    if(iPrintout)
    {
        printf("shfl xor 1\t\t: ");
        printData(h_outData, nElem);
    }

    test_shfl_xor<<<1, block>>>(d_outData, d_inData, -8);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if(iPrintout)
    {
        printf("shfl xor -1\t\t: ");
        printData(h_outData, nElem);
    }

    // shfl xor - int4
    test_shfl_xor_int4<<<1, block / SEGM>>>(d_outData, d_inData, 1);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if(iPrintout)
    {
        printf("initialData\t\t: ");
        printData(h_inData, nElem);
    }

    if(iPrintout)
    {
        printf("shfl int4 1\t\t: ");
        printData(h_outData, nElem);
    }

    // shfl xor - register array
    test_shfl_xor_array<<<1, block / SEGM>>>(d_outData, d_inData, 1);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if(iPrintout)
    {
        printf("initialData\t\t: ");
        printData(h_inData, nElem);
    }

    if(iPrintout)
    {
        printf("shfl array 1\t\t: ");
        printData(h_outData, nElem);
    }

    // shfl xor - test_shfl_xor_element
    test_shfl_xor_element<<<1, block / SEGM>>>(d_outData, d_inData, 1, 0, 3);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if(iPrintout)
    {
        printf("initialData\t\t: ");
        printData(h_inData, nElem);
    }

    if(iPrintout)
    {
        printf("shfl idx \t\t: ");
        printData(h_outData, nElem);
    }

    // shfl xor - swap
    test_shfl_xor_array_swap_base<<<1, block / SEGM>>>(d_outData, d_inData, 1,
            0, 3);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if(iPrintout)
    {
        printf("shfl swap base\t\t: ");
        printData(h_outData, nElem);
    }

    // shfl xor - swap
    test_shfl_xor_array_swap<<<1, block / SEGM>>>(d_outData, d_inData, 1, 0, 3);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if(iPrintout)
    {
        printf("shfl swap 0 3\t\t: ");
        printData(h_outData, nElem);
    }

    // shfl xor - swap
    test_shfl_swap<<<1, block / SEGM>>>(d_outData, d_inData, 1, 0, 3);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if(iPrintout)
    {
        printf("shfl swap inline\t: ");
        printData(h_outData, nElem);
    }

    // shfl xor - register array
    test_shfl_array<<<1, block / SEGM>>>(d_outData, d_inData, 1);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if(iPrintout)
    {
        printf("initialData\t\t: ");
        printData(h_inData, nElem);
    }

    if(iPrintout)
    {
        printf("shfl array \t\t: ");
        printData(h_outData, nElem);
    }

    // finishing
    CHECK(cudaFree(d_inData));
    CHECK(cudaFree(d_outData));
    CHECK(cudaDeviceReset();  );

    return EXIT_SUCCESS;
}
