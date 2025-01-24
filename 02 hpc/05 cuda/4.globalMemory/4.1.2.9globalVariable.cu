#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/common.h"

/*
    一个浮点类型的全局变量在文件作用域内被声明。
    在核函数 checkGolbalVariable 中，全局变量的值在输出之后，就发生了改变。
    在主函数中，全局变量的值是通过函数cudaMemcpyToSymbol初始化的。
    在执行完 checkGlobalVariable 函数后，全局变量的值被替换掉了。
    新的值通过使用 cudaMemcpyFromSymbol 函数被复制回主机。
 */

__device__ float devData;

__global__ void checkGlobalVariable()
{
    // display the original value
    printf("Device: the value of the global variable is %f\n", devData);

    // alter the value
    devData += 2.0f;
}

int main(void)
{
    // initialize the global variable
    float value = 3.14f;
    CHECK(cudaMemcpyToSymbol(devData, &value, sizeof(float)));
    printf("Host:   copied %f to the global variable\n", value);

    // invoke the kernel
    checkGlobalVariable<<<1, 1>>>();

    // copy the global variable back to the host
    CHECK(cudaMemcpyFromSymbol(&value, devData, sizeof(float)));
    printf("Host:   the value changed by the kernel to %f\n", value);

    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
