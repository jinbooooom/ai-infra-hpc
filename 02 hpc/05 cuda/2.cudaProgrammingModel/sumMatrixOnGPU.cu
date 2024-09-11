#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * This example demonstrates a simple vector sum on the GPU and on the host.
 * The performance of a variety of CUDA thread configurations is tested using
 * the sumMatrixOnGPU2D, sumMatrixOnGPU1D, and sumMatrixOnGPUMix kernels.
 * sumArraysOnHost sequentially iterates through vector elements on the host.
 */

void initialData(float *ip, const float ival, int size)
{
    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 100.0f;
    }

    return;
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];

        }

        ia += nx;
        ib += nx;
        ic += nx;
    }

    return;
}

void printMatrix(float *C, const int nx, const int ny)
{
    float *ic = C;

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            printf("%f ", ic[ix]);

        }

        ic += nx;
        printf("\n");
    }

    return;
}

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (match)
        printf("Arrays match.\n\n");
    else
        printf("Arrays do not match.\n\n");
}

// grid 2D block 2D
__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int nx,
                                 int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
        MatC[idx] = MatA[idx] + MatB[idx];
}

// grid 1D block 1D
__global__ void sumMatrixOnGPU1D(float *MatA, float *MatB, float *MatC, int nx,
                                 int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;

    if (ix < nx )
        for (int iy = 0; iy < ny; iy++)
        {
            int idx = iy * nx + ix;
            MatC[idx] = MatA[idx] + MatB[idx];
        }


}

// grid 2D block 1D
__global__ void sumMatrixOnGPUMix(float *MatA, float *MatB, float *MatC, int nx,
                                  int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = blockIdx.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
        MatC[idx] = MatA[idx] + MatB[idx];
}


int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up data size of matrix
    int nx = 1 << 14;
    int ny = 1 << 14;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // initialize data at host side
    double iStart = seconds();
    initialData(h_A,  2.0f, nxy);
    initialData(h_B,  0.5f, nxy);
    double iElaps = seconds() - iStart;
    printf("Matrix initialization elapsed %f sec\n", iElaps);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    iStart = seconds();
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    iElaps = seconds() - iStart;
    printf("sumMatrixOnHost elapsed %f sec\n", iElaps);

    float *d_MatA, *d_MatB, *d_MatC;
    CHECK(cudaMalloc((void **)&d_MatA, nBytes));
    CHECK(cudaMalloc((void **)&d_MatB, nBytes));
    CHECK(cudaMalloc((void **)&d_MatC, nBytes));

    CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    iStart = seconds();
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumMatrixOnGPU2D <<<  (%d,%d), (%d,%d)  >>> elapsed %f sec\n",
           grid.x, grid.y, block.x, block.y, iElaps);

    // adjust block size
    block.x = 16;
    grid.x  = (nx + block.x - 1) / block.x;
    grid.y  = (ny + block.y - 1) / block.y;

    iStart = seconds();
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumMatrixOnGPU2D <<<  (%d,%d), (%d,%d)  >>> elapsed %f sec\n",
           grid.x, grid.y, block.x, block.y, iElaps);

    // adjust block size
    block.y = 16;
    block.x = 32;
    grid.x  = (nx + block.x - 1) / block.x;
    grid.y  = (ny + block.y - 1) / block.y;

    iStart = seconds();
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumMatrixOnGPU2D <<<  (%d,%d), (%d,%d)  >>> elapsed %f sec\n",
           grid.x, grid.y, block.x, block.y, iElaps);

    block.y = 16;
    block.x = 16;
    grid.x  = (nx + block.x - 1) / block.x;
    grid.y  = (ny + block.y - 1) / block.y;

    iStart = seconds();
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumMatrixOnGPU2D <<<  (%d,%d), (%d,%d)  >>> elapsed %f sec\n",
           grid.x, grid.y, block.x, block.y, iElaps);

    block.y = 16;
    block.x = 64;
    grid.x  = (nx + block.x - 1) / block.x;
    grid.y  = (ny + block.y - 1) / block.y;

    iStart = seconds();
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumMatrixOnGPU2D <<<  (%d,%d), (%d,%d)  >>> elapsed %f sec\n",
           grid.x, grid.y, block.x, block.y, iElaps);

    block.y = 64;
    block.x = 16;
    grid.x  = (nx + block.x - 1) / block.x;
    grid.y  = (ny + block.y - 1) / block.y;

    iStart = seconds();
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumMatrixOnGPU2D <<<  (%d,%d), (%d,%d)  >>> elapsed %f sec\n",
           grid.x, grid.y, block.x, block.y, iElaps);

    block.x = 32;
    grid.x  = (nx + block.x - 1) / block.x;
    block.y = 1;
    grid.y  = 1;

    iStart = seconds();
    sumMatrixOnGPU1D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumMatrixOnGPU1D <<<  (%d,%d), (%d,%d)  >>> elapsed %f sec\n",
           grid.x, grid.y, block.x, block.y, iElaps);

    block.x = 64;
    grid.x  = (nx + block.x - 1) / block.x;
    block.y = 1;
    grid.y  = 1;

    iStart = seconds();
    sumMatrixOnGPU1D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumMatrixOnGPU1D <<<  (%d,%d), (%d,%d)  >>> elapsed %f sec\n",
           grid.x, grid.y, block.x, block.y, iElaps);

    block.x = 128;
    grid.x  = (nx + block.x - 1) / block.x;
    block.y = 1;
    grid.y  = 1;

    iStart = seconds();
    sumMatrixOnGPU1D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumMatrixOnGPU1D <<<  (%d,%d), (%d,%d)  >>> elapsed %f sec\n",
           grid.x, grid.y, block.x, block.y, iElaps);

    // grid 2D and block 1D
    block.x = 32;
    grid.x  = (nx + block.x - 1) / block.x;
    block.y = 1;
    grid.y  = ny;

    iStart = seconds();
    sumMatrixOnGPUMix<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumMatrixOnGPUMix <<<  (%d,%d), (%d,%d)  >>> elapsed %f sec\n",
           grid.x, grid.y, block.x, block.y, iElaps);

    block.x = 64;
    grid.x  = (nx + block.x - 1) / block.x;
    block.y = 1;
    grid.y  = ny;

    iStart = seconds();
    sumMatrixOnGPUMix<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumMatrixOnGPUMix <<<  (%d,%d), (%d,%d)  >>> elapsed %f sec\n",
           grid.x, grid.y, block.x, block.y, iElaps);

    block.x = 128;
    grid.x  = (nx + block.x - 1) / block.x;
    block.y = 1;
    grid.y  = ny;

    iStart = seconds();
    sumMatrixOnGPUMix<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumMatrixOnGPUMix <<<  (%d,%d), (%d,%d)  >>> elapsed %f sec\n",
           grid.x, grid.y, block.x, block.y, iElaps);

    block.x = 256;
    grid.x  = (nx + block.x - 1) / block.x;
    block.y = 1;
    grid.y  = ny;

    iStart = seconds();
    sumMatrixOnGPUMix<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumMatrixOnGPUMix <<<  (%d,%d), (%d,%d)  >>> elapsed %f sec\n",
           grid.x, grid.y, block.x, block.y, iElaps);

    block.x = 512;
    grid.x  = (nx + block.x - 1) / block.x;
    block.y = 1;
    grid.y  = ny;

    iStart = seconds();
    sumMatrixOnGPUMix<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumMatrixOnGPUMix <<<  (%d,%d), (%d,%d)  >>> elapsed %f sec\n",
           grid.x, grid.y, block.x, block.y, iElaps);

    CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, nxy);

    CHECK(cudaFree(d_MatA));
    CHECK(cudaFree(d_MatB));
    CHECK(cudaFree(d_MatC));

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return (0);
}
