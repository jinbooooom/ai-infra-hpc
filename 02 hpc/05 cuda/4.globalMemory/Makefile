# 定义 CUDA 和 C 应用程序
CU_APPS = 4.1.2.9globalVariable 4.2.2memTransfer 4.2.3pinMemTransfer 4.3.2readSegment \
		4.3.5readSegmentUnroll 4.3.4simpleMathAoS 4.3.4simpleMathSoA 4.2.4sumArrayZerocpy \
		4.5sumMatrixGPUManaged 4.5sumMatrixGPUManual 4.4.2transpose 4.3.3writeSegment
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
