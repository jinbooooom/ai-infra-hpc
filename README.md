# AI-Infra HPC 学习与总结
### 本仓库用于记录 AI-Infra 与 HPC 技术：

- AI System 的底层技术
- 集群多机多卡互联技术
- 并行计算与高性能计算
- 训练与推理

### 文件夹说明

各文件夹的内容如下：

```shell
├── 01 chip		# 芯片硬件
├── 02 hpc		# 高性能计算
│   ├── 01 openmp	
│   ├── 02 simd
│   └── 05 cuda		# GPU 编程
├── 03 link		# 多机多卡互联底层通信
│   ├── 01 noc		# 片上网络
│   ├── 02 pcie		# PCI-Express(peripheral component interconnect express)
│   ├── 03 topo		# 多机多卡互联以及拓扑相关
│   ├── 05 gpuDirect 	# GPU 与 Host、GPU、IB 网卡、NVMe SSD 的底层通信
│   └── 08 infiniband 	# IB 网卡与 RDMA 通信
├── 04 storage		# 存储
├── 05 ccl 		# 集合通信
│   ├── mpi 		# MPI 的使用与教程
│   └── nccl 		# NCCL 的使用、设计
├── 06 train&infer 	# 训练与推理
├── 99 industryReport 	# 行业前沿报告
```

