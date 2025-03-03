## GPUDirect

GPU Direct 是 NVIDIA 开发的一项技术，可实现 GPU 与其他设备（例如IB 网卡 (NIC) 和存储设备）之间的直接通信和数据传输，而不涉及 CPU。

传统上，当数据需要在 GPU 和另一个设备之间传输时，数据必须通过 CPU，从而导致潜在的瓶颈并增加延迟。使用 GPUDirect，网络适配器和存储驱动器可以直接读写 GPU 内存，减少不必要的内存消耗，减少 CPU 开销并降低延迟，从而显著提高性能。GPU Direct 技术包括 GPUDirect  Storage、GPUDirect RDMA、GPUDirect P2P。

### **GDS（GPUDirect Storage）**

DeepSpeed框架 ZeRo-infinity级优化支持将模型存储在NVME中，解决超大模型训练时显存不足的问题。

GPUDirect Storage 允许存储设备和 GPU 之间进行直接数据传输，绕过 CPU，减少数据传输的延迟和 CPU 开销。

通过 GPUDirect Storage，GPU 可以直接从存储设备（如固态硬盘（SSD）或非易失性内存扩展（NVMe）驱动器）访问数据，而无需将数据先复制到 CPU 的内存中。这种直接访问能够实现更快的数据传输速度，并更高效地利用 GPU 资源。

![img](assets/readme/v2-40a61d0d75180f43474c80c7fae50d26_1440w.jpg)

GPUDirect Storage 的主要特点和优势包括：

- 减少 CPU 参与：通过绕过 CPU，实现 GPU 和存储设备之间的直接通信，GPUDirect Storage 减少了 CPU 开销，并释放 CPU 资源用于其他任务，从而改善系统的整体性能。
- 低延迟数据访问：GPUDirect Storage 消除了数据通过 CPU 的传输路径，从而最小化了数据传输的延迟。

提高存储性能：通过允许 GPU 直接访问存储设备，GPUDirect Storage 实现了高速数据传输，可以显著提高存储性能，加速数据密集型工作负载的处理速度。

增强的可扩展性：GPUDirect Storage 支持多 GPU 配置，允许多个 GPU 同时访问存储设备。这种可扩展性对于需要大规模并行处理和数据分析的应用至关重要。

![image-20250102143811067](assets/readme/image-20250102143811067.png)

![image-20250102144025001](assets/readme/image-20250102144025001.png)

![image-20250102144007470](assets/readme/image-20250102144007470.png)

![image-20250102143903609](assets/readme/image-20250102143903609.png)

### **GPUDirect P2P**

某些工作负载需要位于同一服务器中的两个或多个 GPU 之间进行数据交换，在没有 GPUDirect P2P 技术的情况下，来自 GPU 的数据将首先通过 CPU 和 PCIe 总线复制到主机固定的共享内存。然后，数据将通过 CPU 和 PCIe 总线从主机固定的共享内存复制到目标 GPU，数据在到达目的地之前需要被复制两次、

![v2-6a4c295f35cd1ff89030b16aa4f686e6_b](assets/readme/v2-6a4c295f35cd1ff89030b16aa4f686e6_b.gif)

有了 GPUDirect P2P 通信技术后，将数据从源 GPU 复制到同一节点中的另一个 GPU 不再需要将数据临时暂存到主机内存中。如果两个 GPU 连接到同一 PCIe 总线，GPUDirect P2P 允许访问其相应的内存，而无需 CPU 参与。前者将执行相同任务所需的复制操作数量减半。

![v2-a6bccf8e702aaf51d320d747c74c13dc_b](assets/readme/v2-a6bccf8e702aaf51d320d747c74c13dc_b.gif)

### GDR（GPUDirect RDMA）



GPUDirect RDMA 它允许 GPU 直接访问 RDMA 网络设备中的数据，无需通过主机内存或 CPU 。

无 GPUDirect RDMA 的数据传输情况如下：

![MLFrameworksPart2_Pic6](assets/readme/MLFrameworksPart2_Pic6.gif)

启用 GPUDirect RDMA 的数据传输情况如下：![MLFrameworksPart2_Pic7](assets/readme/MLFrameworksPart2_Pic7.gif)

![img](assets/readme/v2-da6ac66f0c2b99d7bc053f6bf42661ae_1440w.jpg)GPUDirect RDMA 通过绕过主机内存和 CPU，直接在 GPU 和 RDMA 网络设备之间进行数据传输，显著降低传输延迟，加快数据交换速度，并可以减轻  CPU 负载，释放 CPU 的计算能力。另外，GPUDirect RDMA 技术允许 GPU 直接访问 RDMA  网络设备中的数据，避免了数据在主机内存中的复制，提高了数据传输的带宽利用率。

#### GPU上的RDMA操作

在使用了GPU Direct RDMA的GPU中，网卡是怎么和GPU配合，实现将GPU的HBM的数据发送到远端的呢？

**在引入InfiniBand GPUDirect Async(IBGDA)之前，是使用CPU上的代理线程来进行网络通信的。NCCL中也有类似的proxy thread和相应实现。**

![image-20250303153817686](assets/readme/image-20250303153817686.png)

Proxy-initiated  communication（https://developer.nvidia.com/blog/improving-network-performance-of-hpc-systems-using-nvidia-magnum-io-nvshmem-and-gpudirect-async/）

此时流程是这样的：

1. 应用程序启动一个CUDA kernel，在GPU内存中产生数据。
2. kernel function通过往CPU memory中的proxy buffer写入数据的方式，通知CPU要进行网络操作。我们将这个通知称为work descriptor, 它包含源地址、目标地址、数据大小及其他必要的网络信息。
3. CPU上的proxy thread收到worker descriptor，并发起相应的网络操作。CPU会更新host memory中的doorbell record (DBR) buffer。（This buffer is used in the recovery path in case the NIC  drops the write to its doorbell.  就是用来记录doorbell的信息，万一硬件来不及及时响应doorbell并把它丢掉，你还能从DBR buffer中恢复doorbell）
4. CPU通过写入NIC的 doorbell (DB)通知NIC。DB是NIC硬件中的一个寄存器。
5. NIC从WQ中读取work descriptor。
6. NIC使用GPUDirect RDMA直接从GPU内存搬运数据。
7. NIC将数据传输到远程节点。
8. NIC通过向主机内存中的CQ写入事件来指示网络操作已完成。
9. CPU轮询CQ以检测网络操作的完成。
10. CPU通知GPU操作已完成。

可以发现，这个过程竟然需要GPU, CPU, NIC三方参与。CPU就像是一个中转站，那么显然它有一些缺点：

- proxy thread消耗了CPU cycles
- proxy thread成为瓶颈，导致在细粒度传输（小消息）时无法达到NIC的峰值吞吐。现代NIC每秒可以处理数亿个通信请求。GPU可以按照该速率生成请求，但CPU的处理速率低得多，造成了在细粒度通信时的瓶颈。

#### InfiniBand GPUDirect Async

IBGDA 是基于 GPUDirect RDMA 技术进一步发展而来的。GPUDirect RDMA 提供了让 GPU  内存能够被直接访问的机制，而 IBGDA 在此基础上，直接在 GPU 内存中生成网络工作描述符（Work  Queue, WQ），并通过 GPUDirect RDMA 通知 InfiniBand 网卡执行数据传输，无需 CPU  干预。其工作队列（WQ）和门铃寄存器（DBR）缓冲区从主机内存迁移到 GPU 内存，减少了 GPU 与 CPU  间的数据复制开销。

IBGDA 能够显著提升细粒度通信的吞吐量，尤其适用于处理每秒数亿级请求的现代 InfiniBand 网卡的场景，在分布式训练中的梯度同步等高频次、小数据量的节点间通信任务中表现出色。

![image-20250303153852214](assets/readme/image-20250303153852214.png)

IBGDA  (https://developer.nvidia.com/blog/improving-network-performance-of-hpc-systems-using-nvidia-magnum-io-nvshmem-and-gpudirect-async)

1. CPU程序启动一个CUDA kernel function，在GPU内存中生成数据。
2. 使用SM创建一个NIC work descriptor，并将其直接写入WQ。与CPU proxy thread不同，该WQ区位于GPU内存中。
3. SM更新DBR buffer，它也位于GPU内存中。
4. SM通过写入NIC的DB寄存器通知NIC。
5. NIC使用GPUDirect RDMA从WQ读取工作描述符。
6. NIC使用GPUDirect RDMA读取GPU内存中的数据。
7. NIC将数据传输到远程节点。
8. NIC通过使用GPUDirect RDMA向CQ缓冲区写入事件，通知GPU网络操作已完成。

可见，IBGDA消除了CPU在通信控制路径中的作用。在使用IBGDA时，GPU和NIC直接交换进行通信所需的信息。WQ和DBR buffer也被移到GPU内存中，以提高SM访问效率。

那么如何能使用上述功能呢？实际上NVIDIA OpenSHMEM Library (NVSHMEM)早已把IBGDA的能力加入进它的库中，并且NVSHMEM 为所有参与计算的 GPU  提供了一个统一的、对称的全局地址空间，方便用户的开发。DeepSeek开源的DeepEP也使用了NVSHMEM。

参考：

- [Improving Network Performance of HPC Systems Using NVIDIA Magnum IO NVSHMEM and GPUDirect Async](https://developer.nvidia.com/blog/improving-network-performance-of-hpc-systems-using-nvidia-magnum-io-nvshmem-and-gpudirect-async/)
- [浅析DeepSeek中提到的IBGDA](https://zhuanlan.zhihu.com/p/26082845081)

## GDRCopy

GDRCopy 是一个基于 GPUDirect RDMA 技术的低延迟 GPU 内存拷贝库，它允许 CPU 直接映射和访问 GPU  内存。GDRCopy 还提供了优化的拷贝 API，并被广泛应用于高性能通信运行时环境中，如 UCX、OpenMPI、MVAPICH 和  NVSHMEM。

![magnum-io-cudaMemcpy-vs-GDRCopy](assets/readme/magnum-io-cudaMemcpy-vs-GDRCopy.svg)

`cudaMemcpy` 利用 GPU 的 DMA 引擎在 CPU 和 GPU 内存之间传输数据，这会触发 DMA  引擎的操作，导致在小数据量时产生延迟开销和性能下降。而 GDRCopy 则允许 CPU 通过 BAR（Base Address  Register）映射直接访问 GPU 内存，从而实现了 GPU 和 CPU 内存之间的低延迟拷贝。

![magnum-io-gdrcopy-h2d-and-d2h](assets/readme/magnum-io-gdrcopy-h2d-and-d2h.svg)

## 参考

- [Machine Learning Frameworks Interoperability, Part 2: Data Loading and Data Transfer Bottlenecks](https://developer.nvidia.com/blog/machine-learning-frameworks-interoperability-part-2-data-loading-and-data-transfer-bottlenecks/)
- [Enable faster memory transfers between CPU and GPU with GDRCopy](https://developer.nvidia.com/gdrcopy)





