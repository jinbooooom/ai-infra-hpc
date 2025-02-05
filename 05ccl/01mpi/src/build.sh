#!/bin/bash

# 脚本名称：build.sh
# 功能：自动执行 CMake 和 Make 构建过程

# 设置构建目录名称
BUILD_DIR="build"
# 设置可执行文件输出目录
BIN_DIR="bin"

# 检查并创建构建目录
if [ ! -d "${BUILD_DIR}" ]; then
    echo "创建构建目录: ${BUILD_DIR}"
    mkdir -p ${BUILD_DIR}
fi

# 进入构建目录
cd ${BUILD_DIR}

# 运行 CMake 生成构建系统
echo "运行 CMake..."
cmake ..

# 检查 CMake 是否成功
if [ $? -ne 0 ]; then
    echo "CMake 失败，请检查错误信息。"
    exit 1
fi

# 运行 Make 编译项目
echo "运行 Make..."
make -j`nproc`

# 检查 Make 是否成功
if [ $? -ne 0 ]; then
    echo "Make 失败，请检查错误信息。"
    exit 1
fi

# 返回项目根目录
cd ..

# 检查并创建 bin 目录
if [ ! -d "${BIN_DIR}" ]; then
    echo "创建可执行文件输出目录: ${BIN_DIR}"
    mkdir -p ${BIN_DIR}
fi

# 提示编译完成
echo "编译完成！可执行文件已保存到 ${BIN_DIR} 目录。"