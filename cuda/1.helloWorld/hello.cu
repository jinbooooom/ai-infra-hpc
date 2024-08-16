#include <stdio.h>

// 定义 CUDA 核函数，修饰符__global__告诉编译器这个函数将会从CPU中调用，然后在GPU上执行
__global__ void helloCUDA()
{
    printf("threadIdx %d, blockIdx %d, blockDim %d: Hello, World!\n", threadIdx.x , blockIdx.x , blockDim.x );
}

int main()
{
    /* 
    调用 CUDA 核函数三重尖括号意味着从主线程到设备端代码的调用。
    一个内核函数通过一组线程来执行，所有线程执行相同的代码。
    三重尖括号里面的参数是执行配置，用来说明使用多少线程来执行内核函数。
    在这个例子中，有10个GPU线程被调用。
    */
    helloCUDA<<<1, 10>>>();
    
    // 等待 GPU 执行完毕
    cudaDeviceSynchronize();
    
    return 0;
}

/*
输出：
threadIdx 0, blockIdx 0, blockDim 10: Hello, World!
threadIdx 1, blockIdx 0, blockDim 10: Hello, World!
threadIdx 2, blockIdx 0, blockDim 10: Hello, World!
threadIdx 3, blockIdx 0, blockDim 10: Hello, World!
threadIdx 4, blockIdx 0, blockDim 10: Hello, World!
threadIdx 5, blockIdx 0, blockDim 10: Hello, World!
threadIdx 6, blockIdx 0, blockDim 10: Hello, World!
threadIdx 7, blockIdx 0, blockDim 10: Hello, World!
threadIdx 8, blockIdx 0, blockDim 10: Hello, World!
threadIdx 9, blockIdx 0, blockDim 10: Hello, World!
*/
