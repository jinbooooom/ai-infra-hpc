#!/bin/bash

# CUDA-GDB 示例程序编译脚本

echo "编译 CUDA-GDB 示例程序..."

# 检查 nvcc 是否可用
if ! command -v nvcc &> /dev/null; then
    echo "错误: 未找到 nvcc，请确保 CUDA 已正确安装并配置 PATH"
    exit 1
fi

# 编译选项说明:
# -g: 为主机代码生成调试信息
# -G: 为设备代码生成调试信息（必须使用 -G 才能调试内核，已包含行号信息）
# -O0: 禁用优化，便于调试
# --ptxas-options=-w: 抑制 ptxas 警告和信息输出
# -Wno-deprecated-gpu-targets: 抑制架构警告（可选）

echo "编译模式: Debug (带调试信息)"
nvcc -g -G -O0 -Wno-deprecated-gpu-targets --ptxas-options=-w \
     example.cu -o example

if [ $? -eq 0 ]; then
    echo "编译成功！"
    echo "可执行文件: ./example"
    echo ""
    echo "运行程序:"
    echo "  ./example"
    echo ""
    echo "使用 cuda-gdb 调试:"
    echo "  cuda-gdb ./example"
else
    echo "编译失败！"
    exit 1
fi

