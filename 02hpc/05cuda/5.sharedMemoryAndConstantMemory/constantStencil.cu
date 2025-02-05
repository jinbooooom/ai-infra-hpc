#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * An example of using constant memory to optimize performance of a stencil
 * computation by storing coefficients of the computation in a constant memory
 * array (coef).
 */

#define RADIUS 4
#define BDIM 32

// constant memory
__constant__ float coef[RADIUS + 1];

// FD coeffecient
#define a0     0.00000f
#define a1     0.80000f
#define a2    -0.20000f
#define a3     0.03809f
#define a4    -0.00357f

void initialData(float *in,  const int size)
{
    for (int i = 0; i < size; i++)
    {
        in[i] = (float)(rand() & 0xFF) / 100.0f;
    }
}

void printData(float *in,  const int size)
{
    for (int i = RADIUS; i < size; i++)
    {
        printf("%f ", in[i]);
    }

    printf("\n");
}

void setup_coef_constant (void)
{
    const float h_coef[] = {a0, a1, a2, a3, a4};
    CHECK(cudaMemcpyToSymbol( coef, h_coef, (RADIUS + 1) * sizeof(float)));
}

void cpu_stencil_1d (float *in, float *out, int isize)
{
    for (int i = RADIUS; i <= isize; i++)
    {
        float tmp = a1 * (in[i + 1] - in[i - 1])
                    + a2 * (in[i + 2] - in[i - 2])
                    + a3 * (in[i + 3] - in[i - 3])
                    + a4 * (in[i + 4] - in[i - 4]);
        out[i] = tmp;
    }
}

void checkResult(float *hostRef, float *gpuRef, const int size)
{
    double epsilon = 1.0E-6;
    bool match = 1;

    for (int i = RADIUS; i < size; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i, hostRef[i],
                   gpuRef[i]);
            break;
        }
    }

    if (!match) printf("Arrays do not match.\n\n");
}

__global__ void stencil_1d(float *in, float *out, int N)
{
    // shared memory
    __shared__ float smem[BDIM + 2 * RADIUS];

    // index to global memory
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (idx < N)
    {

        // index to shared memory for stencil calculatioin
        int sidx = threadIdx.x + RADIUS;

        // Read data from global memory into shared memory
        smem[sidx] = in[idx];

        // read halo part to shared memory
        if (threadIdx.x < RADIUS)
        {
            smem[sidx - RADIUS] = in[idx - RADIUS];
            smem[sidx + BDIM] = in[idx + BDIM];
        }

        // Synchronize (ensure all the data is available)
        __syncthreads();

        // Apply the stencil
        float tmp = 0.0f;

#pragma unroll
        for (int i = 1; i <= RADIUS; i++)
        {
            tmp += coef[i] * (smem[sidx + i] - smem[sidx - i]);
        }

        // Store the result
        out[idx] = tmp;

        idx += gridDim.x * blockDim.x;
    }
}


int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting transpose at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up data size
    int isize = 1 << 24;

    size_t nBytes = (isize + 2 * RADIUS) * sizeof(float);
    printf("array size: %d ", isize);

    bool iprint = 0;

    // allocate host memory
    float *h_in    = (float *)malloc(nBytes);
    float *hostRef = (float *)malloc(nBytes);
    float *gpuRef  = (float *)malloc(nBytes);

    // allocate device memory
    float *d_in, *d_out;
    CHECK(cudaMalloc((float**)&d_in, nBytes));
    CHECK(cudaMalloc((float**)&d_out, nBytes));

    // initialize host array
    initialData(h_in, isize + 2 * RADIUS);

    // Copy to device
    CHECK(cudaMemcpy(d_in, h_in, nBytes, cudaMemcpyHostToDevice));

    // set up constant memory
    setup_coef_constant();

    // launch configuration
    cudaDeviceProp info;
    CHECK(cudaGetDeviceProperties(&info, 0));
    dim3 block(BDIM, 1);
    dim3 grid(info.maxGridSize[0] < isize / block.x ? info.maxGridSize[0] :
            isize / block.x, 1);
    printf("(grid, block) %d,%d \n ", grid.x, block.x);

    // Launch stencil_1d() kernel on GPU
    stencil_1d<<<grid, block>>>(d_in + RADIUS, d_out + RADIUS, isize);

    // Copy result back to host
    CHECK(cudaMemcpy(gpuRef, d_out, nBytes, cudaMemcpyDeviceToHost));

    // apply cpu stencil
    cpu_stencil_1d(h_in, hostRef, isize);

    // check results
    checkResult(hostRef, gpuRef, isize);

    // print out results
    if(iprint)
    {
        printData(gpuRef, isize);
        printData(hostRef, isize);
    }

    // Cleanup
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    free(h_in);
    free(hostRef);
    free(gpuRef);

    // reset device
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
