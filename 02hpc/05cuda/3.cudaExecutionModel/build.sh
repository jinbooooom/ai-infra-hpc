#!/bin/bash

# 默认的 Makefile
DEFAULT_MAKEFILE="Makefile"

# 如果传入参数 "k"，则使用 Makefile_KernelLaunchKernel
if [[ "$1" == "k" ]]; then
    MAKE_FILE="Makefile_KernelLaunchKernel"
else
    MAKE_FILE="$DEFAULT_MAKEFILE"
fi

# 检查指定的 Makefile 是否存在
if [[ ! -f "$MAKE_FILE" ]]; then
    echo "错误: $MAKE_FILE 不存在！"
    exit 1
fi

# 执行 make 命令
echo "正在使用 $MAKE_FILE 进行编译..."
make -f "$MAKE_FILE"