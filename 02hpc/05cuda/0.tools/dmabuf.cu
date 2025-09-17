#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <cstring>
#include <unistd.h>  // for getpagesize()

int main() {
    // Initialize CUDA
    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to initialize CUDA, error: " << result << std::endl;
        return -1;
    }

    // Set device
    cudaError_t cudaResult = cudaSetDevice(0);
    if (cudaResult != cudaSuccess) {
        std::cerr << "Failed to set device, error: " << cudaResult << std::endl;
        return -1;
    }

    // Get page size for proper alignment
    size_t pageSize = getpagesize();
    std::cout << "Host page size: " << pageSize << " bytes" << std::endl;
    
    // Allocate GPU memory aligned to page size
    CUdeviceptr gpuPtr;
    size_t size = pageSize; // Align to page size
    result = cuMemAlloc(&gpuPtr, size);
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to allocate GPU memory, error: " << result << std::endl;
        return -1;
    }

    std::cout << "GPU memory allocated successfully at: 0x" << std::hex << gpuPtr << std::dec << std::endl;

    // Check if device supports DMA buffer before attempting to use it
    int dmaBufSupported = 0;
    result = cuDeviceGetAttribute(&dmaBufSupported, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, 0);
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to get DMA buffer support attribute, error: " << result << std::endl;
        cuMemFree(gpuPtr);
        return -1;
    }
    
    std::cout << "Device DMA buffer support: " << (dmaBufSupported ? "YES" : "NO") << std::endl;

    // Try to get DMA buffer fd
    int dmabufFd = -1;
    result = cuMemGetHandleForAddressRange(&dmabufFd,
                                          gpuPtr,
                                          size,
                                          CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
                                          0);
    
    if (result == CUDA_SUCCESS) {
        std::cout << "DMA buffer fd: " << dmabufFd << std::endl;
    } else {
        const char* errorString;
        cuGetErrorString(result, &errorString);
        std::cout << "Failed to get DMA buffer fd, error: " << result << ", " << errorString << std::endl;
    }

    // Clean up
    cuMemFree(gpuPtr);

    return 0;
}

/*
nvcc -o dmabuf dmabuf.cu -lcuda -std=c++11
./dmabuf
*/
