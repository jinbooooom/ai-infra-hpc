#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/common.h"

/*
 * 本示例展示了在矩阵转置核函数中应用的各种内存访问模式优化。
 */

#define BDIMX 16
#define BDIMY 16

void initialData(float *in, const int size)
{
    for (int i = 0; i < size; i++)
    {
        in[i] = (float)(rand() & 0xFF) / 10.0f;  // 100.0f;
    }

    return;
}

void printData(float *in, const int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%dth element: %f\n", i, in[i]);
    }

    return;
}

void checkResult(float *hostRef, float *gpuRef, const int size, int showme)
{
    double epsilon = 1.0E-8;
    bool match     = 1;

    for (int i = 0; i < size; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i, hostRef[i], gpuRef[i]);
            break;
        }

        if (showme && i > size / 2 && i < size / 2 + 5)
        {
            // printf("%dth element: host %f gpu %f\n",i,hostRef[i],gpuRef[i]);
        }
    }

    if (!match)
        printf("Arrays do not match.\n\n");
}

void transposeHost(float *out, float *in, const int nx, const int ny)
{
    for (int iy = 0; iy < ny; ++iy)
    {
        for (int ix = 0; ix < nx; ++ix)
        {
            out[ix * ny + iy] = in[iy * nx + ix];
        }
    }
}

__global__ void warmup(float *out, float *in, const int nx, const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        out[iy * nx + ix] = in[iy * nx + ix];
    }
}

// case 0 copy kernel: access data in rows
__global__ void copyRow(float *out, float *in, const int nx, const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        out[iy * nx + ix] = in[iy * nx + ix];
    }
}

// case 1 copy kernel: access data in columns
__global__ void copyCol(float *out, float *in, const int nx, const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        out[ix * ny + iy] = in[ix * ny + iy];
    }
}

// case 2 transpose kernel: read in rows and write in columns
__global__ void transposeNaiveRow(float *out, float *in, const int nx, const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        out[ix * ny + iy] = in[iy * nx + ix];
    }
}

// case 3 transpose kernel: read in columns and write in rows
__global__ void transposeNaiveCol(float *out, float *in, const int nx, const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        out[iy * nx + ix] = in[ix * ny + iy];
    }
}

// case 4 transpose kernel: read in rows and write in columns + unroll 4 blocks
__global__ void transposeUnroll4Row(float *out, float *in, const int nx, const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    unsigned int ti = iy * nx + ix;  // access in rows
    unsigned int to = ix * ny + iy;  // access in columns

    if (ix + 3 * blockDim.x < nx && iy < ny)
    {
        out[to]                       = in[ti];
        out[to + ny * blockDim.x]     = in[ti + blockDim.x];
        out[to + ny * 2 * blockDim.x] = in[ti + 2 * blockDim.x];
        out[to + ny * 3 * blockDim.x] = in[ti + 3 * blockDim.x];
    }
}

// case 5 transpose kernel: read in columns and write in rows + unroll 4 blocks
__global__ void transposeUnroll4Col(float *out, float *in, const int nx, const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    unsigned int ti = iy * nx + ix;  // access in rows
    unsigned int to = ix * ny + iy;  // access in columns

    if (ix + 3 * blockDim.x < nx && iy < ny)
    {
        out[ti]                  = in[to];
        out[ti + blockDim.x]     = in[to + blockDim.x * ny];
        out[ti + 2 * blockDim.x] = in[to + 2 * blockDim.x * ny];
        out[ti + 3 * blockDim.x] = in[to + 3 * blockDim.x * ny];
    }
}

/*
 * case 6 :  transpose kernel: read in rows and write in colunms + diagonal
 * coordinate transform
 */
__global__ void transposeDiagonalRow(float *out, float *in, const int nx, const int ny)
{
    unsigned int blk_y = blockIdx.x;
    unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;

    unsigned int ix = blockDim.x * blk_x + threadIdx.x;
    unsigned int iy = blockDim.y * blk_y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        out[ix * ny + iy] = in[iy * nx + ix];
    }
}

/*
 * case 7 :  transpose kernel: read in columns and write in row + diagonal
 * coordinate transform.
 */
__global__ void transposeDiagonalCol(float *out, float *in, const int nx, const int ny)
{
    unsigned int blk_y = blockIdx.x;
    unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;

    unsigned int ix = blockDim.x * blk_x + threadIdx.x;
    unsigned int iy = blockDim.y * blk_y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        out[iy * nx + ix] = in[ix * ny + iy];
    }
}

// main functions
int main(int argc, char **argv)
{
    // set up device
    int dev = getGPUId();
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting transpose at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up array size 2048
    int nx = 1 << 11;
    int ny = 1 << 11;

    // select a kernel and block size
    int iKernel = 0;
    int blockx  = 16;
    int blocky  = 16;

    if (argc > 1)
        iKernel = atoi(argv[1]);

    if (argc > 2)
        blockx = atoi(argv[2]);

    if (argc > 3)
        blocky = atoi(argv[3]);

    if (argc > 4)
        nx = atoi(argv[4]);

    if (argc > 5)
        ny = atoi(argv[5]);

    printf(" with matrix nx %d ny %d with kernel %d\n", nx, ny, iKernel);
    size_t nBytes = nx * ny * sizeof(float);

    // execution configuration
    dim3 block(blockx, blocky);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // allocate host memory
    float *h_A     = (float *)malloc(nBytes);
    float *hostRef = (float *)malloc(nBytes);
    float *gpuRef  = (float *)malloc(nBytes);

    // initialize host array
    initialData(h_A, nx * ny);

    // transpose at host side
    transposeHost(hostRef, h_A, nx, ny);

    // allocate device memory
    float *d_A, *d_C;
    CHECK(cudaMalloc((float **)&d_A, nBytes));
    CHECK(cudaMalloc((float **)&d_C, nBytes));

    // copy data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    // warmup to avoide startup overhead
    double iStart = seconds();
    warmup<<<grid, block>>>(d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    double iElaps = seconds() - iStart;
    printf("warmup         elapsed %f sec\n", iElaps);
    CHECK(cudaGetLastError());

    // kernel pointer and descriptor
    void (*kernel)(float *, float *, int, int);
    char *kernelName;

    // set up kernel
    switch (iKernel)
    {
        case 0:
            kernel     = &copyRow;
            kernelName = (char *)"CopyRow       ";
            break;

        case 1:
            kernel     = &copyCol;
            kernelName = (char *)"CopyCol       ";
            break;

        case 2:
            kernel     = &transposeNaiveRow;
            kernelName = (char *)"NaiveRow      ";
            break;

        case 3:
            kernel     = &transposeNaiveCol;
            kernelName = (char *)"NaiveCol      ";
            break;

        case 4:
            kernel     = &transposeUnroll4Row;
            kernelName = (char *)"Unroll4Row    ";
            grid.x     = (nx + block.x * 4 - 1) / (block.x * 4);
            break;

        case 5:
            kernel     = &transposeUnroll4Col;
            kernelName = (char *)"Unroll4Col    ";
            grid.x     = (nx + block.x * 4 - 1) / (block.x * 4);
            break;

        case 6:
            kernel     = &transposeDiagonalRow;
            kernelName = (char *)"DiagonalRow   ";
            break;

        case 7:
            kernel     = &transposeDiagonalCol;
            kernelName = (char *)"DiagonalCol   ";
            break;
    }

    // run kernel
    iStart = seconds();
    kernel<<<grid, block>>>(d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    // calculate effective_bandwidth
    float ibnd = 2 * nx * ny * sizeof(float) / 1e9 / iElaps;
    printf(
        "%s elapsed %f sec <<< grid (%d,%d) block (%d,%d)>>> effective "
        "bandwidth %f GB\n",
        kernelName, iElaps, grid.x, grid.y, block.x, block.y, ibnd);
    CHECK(cudaGetLastError());

    // check kernel results
    if (iKernel > 1)
    {
        CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
        checkResult(hostRef, gpuRef, nx * ny, 1);
    }

    // free host and device memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_C));
    free(h_A);
    free(hostRef);
    free(gpuRef);

    // reset device
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
