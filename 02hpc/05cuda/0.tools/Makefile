# 编译CUDA程序并生成到build目录

# 编译器设置
NVCC = nvcc
CXX = g++

# 编译选项
NVCC_FLAGS = -std=c++11 -O2 -lcuda
CXX_FLAGS = -std=c++11 -O2

# 目录设置
SRC_DIR = .
BUILD_DIR = build
TARGET = $(BUILD_DIR)/dmabuf

# 源文件
SOURCES = dmabuf.cu

# 默认目标
all: $(TARGET)

# 创建build目录
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# 编译目标
$(TARGET): $(SOURCES) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<
	@echo "编译完成: $@"

# 清理
clean:
	rm -rf $(BUILD_DIR)

# 强制重新编译
rebuild: clean all

# 强制编译（不检查时间戳）
force:
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SOURCES)
	@echo "强制编译完成: $(TARGET)"

# 运行程序
run: $(TARGET)
	./$(TARGET)

# 显示帮助
help:
	@echo "可用的目标:"
	@echo "  all     - 编译程序 (默认)"
	@echo "  clean   - 清理build目录"
	@echo "  rebuild - 重新编译"
	@echo "  force   - 强制编译（忽略时间戳）"
	@echo "  run     - 编译并运行程序"
	@echo "  help    - 显示此帮助信息"

# 声明伪目标
.PHONY: all clean rebuild force run help
