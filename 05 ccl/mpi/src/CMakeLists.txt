# 设置 CMake 的最低版本要求
cmake_minimum_required(VERSION 3.10)

# 设置项目名称和版本
project(MultiExecutables VERSION 1.0)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 查找 MPI 库
find_package(MPI REQUIRED)

# 设置可执行文件的输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# 获取当前目录下所有的 .cpp 文件，每个 cpp 对应一个可执行程序
file(GLOB SOURCES "*.cpp")

# 遍历源文件列表，为每个源文件生成一个可执行文件
foreach(source ${SOURCES})
    # 获取文件名（不带扩展名）
    get_filename_component(executable_name ${source} NAME_WE)

    # 添加可执行文件
    add_executable(${executable_name} ${source})

    # 包含 MPI 的头文件目录
    target_include_directories(${executable_name} PRIVATE ${MPI_INCLUDE_PATH})

    # 链接 MPI 库
    target_link_libraries(${executable_name} PRIVATE ${MPI_CXX_LIBRARIES})

    # 设置调试标志
    target_compile_options(${executable_name} PRIVATE -g)
endforeach()