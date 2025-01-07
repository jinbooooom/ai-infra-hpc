APPS=hello

# 定义默认的构建目录，并使用abspath函数确保其为绝对路径 
BUILDDIR ?= build 
override BUILDDIR := $(abspath $(BUILDDIR))

all: dirs ${APPS}

NVCC ?= /usr/local/cuda/bin/nvcc
NVCC_GENCODE ?= -gencode=arch=compute_60,code=sm_60 \
                -gencode=arch=compute_61,code=sm_61 \
                -gencode=arch=compute_70,code=sm_70 \
                -gencode=arch=compute_80,code=sm_80 \
                -gencode=arch=compute_80,code=compute_80

%: %.cu
	$(NVCC) $(NVCC_GENCODE) -O2 -std=c++14 -o $(BUILDDIR)/$@ $<
clean:
	rm -rf $(BUILDDIR)/*

# 创建构建目录的规则 
dirs: # 伪目标名称可以任意取，这里用dirs表示创建目录的步骤 
	mkdir -p $(BUILDDIR)
