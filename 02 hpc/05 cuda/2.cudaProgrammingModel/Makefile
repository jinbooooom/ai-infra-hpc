CU_APPS=checkDeviceInfor checkThreadIndex sumArraysOnGPU-timer \
        sumMatrixOnGPU-1D-grid-1D-block sumMatrixOnGPU-2D-grid-2D-block \
        checkDimension defineGridBlock sumArraysOnGPU-small-case \
        sumMatrixOnGPU-2D-grid-1D-block sumMatrixOnGPU
C_APPS=sumArraysOnHost

# 定义默认的构建目录，并使用abspath函数确保其为绝对路径 
BUILDDIR ?= build 
override BUILDDIR := $(abspath $(BUILDDIR))
 
all: dirs ${C_APPS} ${CU_APPS}

NVCC_GENCODE ?= -gencode=arch=compute_60,code=sm_60 \
                -gencode=arch=compute_61,code=sm_61 \
                -gencode=arch=compute_70,code=sm_70 \
                -gencode=arch=compute_80,code=sm_80 \
                -gencode=arch=compute_80,code=compute_80

%: %.cu
	/usr/local/cuda/bin/nvcc $(NVCC_GENCODE) -O2 -std=c++14 -o $(BUILDDIR)/$@ $<
%: %.c
	gcc -O2 -std=c99 -o $(BUILDDIR)/$@ $<
clean:
	rm -rf $(BUILDDIR)/*

# 创建构建目录的规则 
dirs: # 伪目标名称可以任意取，这里用dirs表示创建目录的步骤 
	mkdir -p $(BUILDDIR)



