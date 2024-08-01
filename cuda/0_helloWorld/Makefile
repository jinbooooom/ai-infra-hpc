APPS=hello

all: ${APPS}

NVCC_GENCODE ?= -gencode=arch=compute_60,code=sm_60 \
                -gencode=arch=compute_61,code=sm_61 \
                -gencode=arch=compute_70,code=sm_70 \
                -gencode=arch=compute_80,code=sm_80 \
                -gencode=arch=compute_80,code=compute_80

%: %.cu
	/usr/local/cuda/bin/nvcc $(NVCC_GENCODE) -O2 -std=c++14 -o $@ $<
clean:
	rm -f ${APPS}
