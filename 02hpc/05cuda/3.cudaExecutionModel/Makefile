# 定义 CUDA 和 C 应用程序
# 单独写一个 CU_K_APPS，是因为涉及到 kernel 启动另一个 kernel，改变了编译选项 
CU_K_APPS = 3.6.2nestedHelloWorld 3.6.3nestedReduce 3.6.3nestedReduce2 3.6.3nestedReduceNosync
CU_APPS = 3.4.2reduceInteger 3.2.5simpleDeviceQuery 3.2.2simpleDivergence 3.2.7sumMatrix
C_APPS =

# 定义默认的构建目录，并使用 abspath 函数确保其为绝对路径
BUILDDIR ?= build
override BUILDDIR := $(abspath $(BUILDDIR))

# 默认目标
all: dirs ${CU_APPS:%=${BUILDDIR}/%} ${C_APPS:%=${BUILDDIR}/%}

# 定义 NVCC 编译器和动态生成 NVCC_GENCODE
NVCC ?= /usr/lib/nvidia-cuda-toolkit/bin/nvcc

# 获取当前 CUDA 版本
CUDA_VERSION := $(shell $(NVCC) --version | grep -oP 'release \K[0-9]+\.[0-9]+')

# 根据 CUDA 版本动态生成 NVCC_GENCODE
ifeq ($(shell echo "$(CUDA_VERSION) >= 12.0" | bc), 1)
    NVCC_GENCODE := -gencode=arch=compute_60,code=sm_60 \
                    -gencode=arch=compute_70,code=sm_70 \
                    -gencode=arch=compute_80,code=sm_80 \
                    -gencode=arch=compute_86,code=sm_86
else ifeq ($(shell echo "$(CUDA_VERSION) >= 11.0" | bc), 1)
    NVCC_GENCODE := -gencode=arch=compute_60,code=sm_60 \
                    -gencode=arch=compute_70,code=sm_70 \
                    -gencode=arch=compute_80,code=sm_80
else ifeq ($(shell echo "$(CUDA_VERSION) >= 10.0" | bc), 1)
    NVCC_GENCODE := -gencode=arch=compute_60,code=sm_60 \
                    -gencode=arch=compute_70,code=sm_70
else
    NVCC_GENCODE := -gencode=arch=compute_60,code=sm_60
endif

# $(NVCC) $(NVCC_GENCODE) -O2 -std=c++14 -dc -o $@ $< -lcudadevrt --relocatable-device-code true
# CUDA 应用程序的编译规则
${BUILDDIR}/%: %.cu
	$(NVCC) -O2 -std=c++14 -o $@ $<

# C 应用程序的编译规则
${BUILDDIR}/%: %.cpp
	g++ -O2 -std=c++14 -o $@ $<

# 清理构建目录
clean:
	rm -rf $(BUILDDIR)/*

# 创建构建目录的规则
dirs:
	mkdir -p $(BUILDDIR)

# 防止将 dirs 视为文件目标
.PHONY: all clean dirs