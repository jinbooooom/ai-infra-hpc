





## [卷积](https://zhuanlan.zhihu.com/p/77471866)

![v2-c14af9d136b1431018146118492b0856_b](./assets/readme/v2-c14af9d136b1431018146118492b0856_b.webp)

## [NLP领域中的token和tokenization到底指的是什么？](https://www.zhihu.com/question/64984731/answer/3492419541)

所以了解 token 是什么，在如今的自然语言处理中，不仅是理论的需要，也是应用的需要。我们小时候学语文，都是字组成词，词连成句，句连成文。而 token 可以是上述的任何一个单位，或者是其中某一个部分。也就是：**字、词、短语、句子**等。而**英文**里面就不太一样了，可能是：**字母、单词、单词的子词、句子**等。

Token对应着文本中的一个元素，通过Tokenization将文本划分成一个个的Token。

**word（词）粒度**

在英文语系中，word（词）级别分词实现很简单，因为有天然的分隔符。在中文里面word（词）粒度，需要一些分词工具比如[jieba](https://zhida.zhihu.com/search?content_id=608942506&content_type=Answer&match_order=1&q=jieba&zhida_source=entity)

，以下是中文和英文的例子：

```text
中文句子：我喜欢看电影和读书。
分词结果：我 | 喜欢 | 看 | 电影 | 和 | 读书。
英文句子：I enjoy watching movies and reading books.
分词结果：I | enjoy | watching | movies | and | reading | books.
```

word（词）粒度的优点有：

- **语义明确**：以词为单位进行分词可以更好地保留每个词的语义，使得文本在后续处理中能够更准确地表达含义。
- **上下文理解**：以词为粒度进行分词有助于保留词语之间的关联性和上下文信息，从而在语义分析和理解时能够更好地捕捉句子的意图。

缺点：

- **长尾效应和稀有词问题**： 词表可能变得巨大，包含很多不常见的词汇，增加存储和训练成本，稀有词的训练数据有限，难以获得准确的表示。
- **OOV（Out-of-Vocabulary）**： 词粒度分词模型只能使用词表中的词来进行处理，无法处理词表之外的词汇，这就是所谓的OOV问题。
- **形态关系和词缀关系**： 无法捕捉同一词的不同形态，也无法有效学习词缀在不同词汇之间的共通性，限制了模型的语言理解能力，比如love和loves在word（词）粒度的词表中将会是两个词。

**2.char（字符）粒度**

以字符为单位进行分词，即将文本拆分成一个个单独的字符作为最小基本单元，这种字符粒度的分词方法适用于多种语言，无论是英文、中文还是其他不同语言，都能够一致地使用字符粒度进行处理，因为英文就26个字母以及其他的一些符号，中文常见字就6000个左右。

```text
中文句子：我喜欢看电影和读书。
分词结果：我 | 喜 | 欢 | 看 | 电 | 影 | 和 | 读 | 书 | 。

英文句子：I enjoy watching movies and reading books.
分词结果：I |   | e | n | j | o | y |   | w | a | t | c | h | i | n | g |   | m | o | v | i | e | s |   | a | n | d |   | r | e | a | d | i | n | g |   | b | o | o | k | s | .
```

char（字符）粒度的优点有：

- **统一处理方式**：字符粒度分词方法适用于不同语言，无需针对每种语言设计不同的分词规则或工具，具有通用性。
- **解决OOV问题**：由于字符粒度分词可以处理任何字符，无需维护词表，因此可以很好地处理一些新创词汇、专有名词等问题。

缺点：

- **语义信息不明确**：字符粒度分词无法直接表达词的语义，可能导致在一些语义分析任务中效果较差。
- **处理效率低**：由于文本被拆分为字符，处理的粒度较小，增加后续处理的计算成本和时间。

**3.subword（子词）粒度**

在很多情况下，既不希望将文本切分成单独的词（太大），也不想将其切分成单个字符（太小），而是希望得到介于词和字符之间的子词单元。这就引入了 subword（子词）粒度的分词方法。

参考：https://www.zhihu.com/question/64984731/answer/3492419541

### **什么是Tokenizer？**

Tokenizer是将文本切分成多个tokens的工具或算法。它负责将原始文本分割成tokens 序列。在NLP中，有多种不同类型的tokenizer，每种tokenizer都有其特定的应用场景和适用范围。

1. **基于字符的Tokenizer**：将文本按照字符分割成token，适用于处理中文等没有空格分隔的语言。但是，正如 GPT-3.5 的切分效果，现在的大模型并不一定会按照这个方式划分，但是但是 [bert-base-chinese](https://zhida.zhihu.com/search?content_id=665058060&content_type=Answer&match_order=1&q=bert-base-chinese&zhida_source=entity) 还是按照这个规则进行划分的。
2. **基于词的Tokenizer**：将文本按照语言的语法规则分割成单词。适用于大部分语言，但对于某些复合词语言效果可能不佳。而最简单的语法规则应该是基于空格分割，它将文本字符串按照空白字符（如空格、制表符、换行符等）进行分割。这种方法适用于英文等使用空格分隔单词的语言，但在处理中文、日语等不使用空格分隔单词的语言时效果不佳。
3. 基于句子的Tokenizer：将文本按照句子进行划分。但是这种在实际应用中并不多见。
4. 基于深度学习的Tokenizer：利用神经网络模型来学习文本字符串的最佳分割方式。这种方法通常使用大量的标注数据进行训练，从而让模型能够捕捉到文本中的复杂特征和规律。基于深度学习的 `Tokenization` 在处理中文等不使用空格分隔单词的语言时表现出色，因为它可以学习到单词和句子的语义信息。

## [大模型参数量和显存的换算关系](https://www.zhihu.com/question/612046818/answer/3438795176)

### **1B参数对应多少G显存？**

B和G都是十亿（1000M或1024M）的意思，M是100万的意思，平时说模型参数有**x**B就是说有**x**十亿个参数，平时说显存有多少G/M是说有多少G/M个**字节**（byte），1个字节=8比特（bit），那么，1B模型参数对应多少G内存和参数的精度有关，如果是全精度训练（fp32），一个参数对应32比特，也就是4个字节，参数换算到显存的时候要乘4，也就是1B模型参数对应4G显存，如果是fp16或者bf16就是乘2，1B模型参数对应2G显存。

### 训练时的显存开销

除了模型参数本身外，训练时的显存开销还有这几个部分：

- 梯度：一个参数对应一个梯度值，所以梯度所占显存是参数的1倍
- 优化器状态：取决于优化器的具体类型，如果是裸SGD就不需要额外显存开销，如果是带一阶动量（momentum）的SGD就是1倍，如果是Adam的话就要在momentum的基础上加上二阶动量，优化器状态所占显存就是参数的2倍。

小结：假设我们全参数微调训练一个参数量为1b(十亿参数)的大模型，优化器为Adam，精度为fp32，忽略数据和hidden states部分的显存占用，那么显存占用为：参数的4G+梯度的4G+优化器状态的8G，共16G。如果是bf16精度训练则要减半，就是8G。如果是混合精度训练则根据各部分的精度调整计算过程。



按照Zomi PPT 中的说法：大模型混合精度训练过程中，使用 BF16 进行前向传递，FP32 反向传递梯度信息；优化器更新模型参数时，使用 FP32 优化器状态、FP32 的梯度来更新模型参数。

在一次训练迭代中，每个可训练参数都对应 1 个梯度，2个优化器状态（Adam）。设模型参数量为 φ(FP16)，那么梯度的参数量为  2φ(FP32)，Adam 优化器的参数量为 4φ(FP32)。

训练总内存 = 模型内存(φ) + 梯度内存(2φ) + 优化器内存(4φ) + 激活内存(Xφ) + 其他内存(1.X φ)

#### 激活值的数据占了显存的大头

LLAMA-13B 模型权重为 25GB，8倍为200GB，200GB/64GB≈3.2，理论上可以放在单机八卡的一个 NPU 节点。为什么 LLAMA-13B 一般最小资源需要两个节点 16卡？

![image-20250318102003888](assets/readme/image-20250318102003888.png)

![image-20250318102026582](assets/readme/image-20250318102026582.png)



### 推理时的显存开销

神经网络推理（Inference）阶段，没有优化器状态和梯度信息，也不需要保存中间激活。因此推理阶段占用的显存要远远远小于训练阶段。

模型推理阶段，占用显存大头是模型参数，一次推理过程中，每个可训练参数都对应 1 个梯度，设模型参数量为 φ，使用 float16 来进行推理，模型参数占用的显存大概是  2φ bytes 。

训练总内存 = 模型内存(φ) + 其他内存(0.X φ)

## 目前训练超大规模语言模型技术路线：GPU + PyTorch + Megatron-LM + DeepSpeed。这几个工具在训练中各自的作用是什么？

|             | DeepSeed | DeepSeed代表性功能                | Megatron | Megatron代表性功能 | 备注                                                         |
| :---------: | -------- | --------------------------------- | -------- | ------------------ | ------------------------------------------------------------ |
| GPU底层优化 | 有       | 开创性的全栈 GPU 内核设计FP6 量化 | 更牛逼   | Fused CUDA Kernels | 毕竟Megatron是Nvidia亲儿子，底层优化信手拈来。               |
|  数据并行   | 更牛逼   | Zero系列的分布式数据并行方案      | 有       | 优化器分片         | Megatron 也做了类似 Zero1 的优化器分片，但数据并行没有 deepspeed 强 |
|  模型并行   | 有       |                                   | 更牛逼   |                    | Megatron的张量并行很牛                                       |



 `PyTorch` 提供动态图机制和自动微分（Autograd），是模型定义、数据处理和训练流程编排的核心框架。

- **动态计算图**：支持复杂控制流（如条件分支、循环），便于调试与实验。
- **分布式接口**：提供`torch.distributed` 模块（如AllReduce通信原语）和`DistributedDataParallel`（DDP）封装。

**与深度优化的结合**

- **扩展接口**：通过`torch.nn.Module` 和`torch.autograd.Function` 允许用户自定义并行策略。
- **生态兼容**：与Megatron-LM和DeepSpeed的API深度集成，例如通过`deepspeed.initialize()` 无缝加载优化器。

`Megatron-LM`：模型并行的核心引擎

1. **核心作用**
    由NVIDIA开发的Megatron-LM专注于**模型并行**，通过张量并行（Tensor Parallelism）和流水线并行（Pipeline Parallelism）拆分超大规模模型。
   - **张量并行**：将单个矩阵运算（如GEMM）按行或列拆分到多卡，例如将Transformer层的MLP和注意力头分布到不同GPU。
   - **流水线并行**：将模型按层拆分（如将前10层分配至GPU 1，后10层至GPU 2），通过微批次（Microbatches）隐藏通信开销。
2. **关键技术细节**
   - **通信优化**：在张量并行中，使用AllReduce同步梯度；在流水线并行中，通过气泡（Bubble）压缩技术减少空闲时间。
   - **显存效率**：结合激活检查点（Activation Checkpointing），仅保留关键中间结果，降低显存占用。

`DeepSpeed`：内存与训练效率的终极优化

1. **核心作用**
    DeepSpeed（微软开发）通过ZeRO（Zero Redundancy Optimizer）系列技术优化显存占用和通信效率，同时提供分布式训练工具链。
   - ZeRO阶段1/2/3
     - **阶段1**：拆分优化器状态（Optimizer States）到多卡，显存降低4倍。
     - **阶段2**：进一步拆分梯度（Gradients），显存降低8倍。
     - **阶段3**：拆分模型参数（Parameters），显存降低至理论极限（与GPU数量成反比）。
   - **混合精度与卸载**：支持FP16/BF16训练，结合CPU/NVMe Offloading，扩展模型规模至万亿参数。
2. **附加功能**
   - **3D并行**：整合数据并行（ZeRO）、模型并行（Megatron-LM）和流水线并行，支持超线性扩展。
   - **通信压缩**：通过梯度稀疏化（如1-bit Adam）减少通信数据量。
   - **容错训练**：自动保存检查点（Checkpoint），支持训练中断恢复。



## 并行

### 数据并行

这是日常应用比较多的情况。每一个device上会加载一份模型，然后把数据分发到每个device并行进行计算，加快训练速度。

常用的 API：

- torch.nn.DataParallel(DP)

- torch.nn.DistributedDataParallel(DDP)

DP 相比 DDP 使用起来更友好（代码少），DDP 支持多机多卡，训练速度更快，负载相对均衡。

#### DP

 ![dp](./assets/readme/dp.png)

![dp2](./assets/readme/dp2.png)



DP多采用参数服务器这一编程框架，一般由若个计算Worker和1个梯度聚合Server组成。Server与每个Worker通讯，Worker间并不通讯。因此Server承担了系统所有的通讯压力。基于此DP常用于单机多卡场景。

在DP中，每个 GPU 上都拷贝一份完整的模型，每个GPU上处理batch的一部分数据，所有GPU上传自己的梯度到server，由server进行累加后，再scatter到各GPU用于更新参数。

如上图所示：

1. 从锁页内存里获取一个batch的数据到 GPU0，GPU0 持有一个模型，其它的GPU有一个过时的模型副本
2. 将一个 batch 数据从主 GPU 分发到所有 GPU 上（每个GPU上持有 batch 的一部分）；
3. 将 model 从主 GPU 分发到所有 GPU 上（每个GPU拥有完整的模型）；
4. 每个 GPU 分别独立进行前向传播，得到 outputs；
5. gather 所有的 outputs 到主GPU，在主 GPU 上，计算损失；
6. Scatter 损失到所有 worker GPU上，反向传播，计算参数梯度；
7. AllReduce 计算梯度；(上图中写的是Reduce，理论上这里应该是 AllReduce)
8. 通过梯度更新模型权重；（经过AllReduce 后，梯度是一样的，而旧模型也是一样的，因此，此时所有的模型更新后也是一样的）

API 如下：

 ```python
 torch.nn.DataParallel(module, device_ids=None, output_device=None, dim=0)
 ```

优缺点

优点：只需要一行代码的增加，易于项目原型的开发

```python
net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
```

缺点：

- 每个前向过程中模型复制引入的延迟
- GPU的利用不均衡
- 不支持多机多卡

##### DP通讯瓶颈与梯度异步更新

DP的框架理解起来不难，但实战中确有两个主要问题：

- **存储开销大**。每块GPU上都存了一份完整的模型，造成冗余。关于这一点的优化，通过ZeRO解决。
- **通讯开销大**。Server需要和每一个Worker进行梯度传输。当Server和Worker不在一台机器上时，Server的带宽将会成为整个系统的计算效率瓶颈。

在DP中任务是串行的，为了解决串行的问题，引入梯度异步更新这一管理层略：

![image-20250318004302006](./assets/readme/image-20250318004302006.png)

上图刻画了在**梯度异步更新**的场景下，某个Worker的计算顺序为：

- 在第10轮计算中，该Worker正常计算梯度，并向Server发送push&pull梯度请求。
- 但是，该Worker并不会实际等到把聚合梯度拿回来，更新完参数W后再做计算。而是直接拿旧的W，吃新的数据，继续第11轮的计算。**这样就保证在通讯的时间里，Worker也在马不停蹄做计算，提升计算通讯比。**
- 当然，异步也不能太过份。只计算梯度，不更新权重，那模型就无法收敛。图中刻画的是**延迟为1**的异步更新，也就是在开始第12轮的计算时，必须保证W已经用第10、11轮的梯度做完2次更新了。

参数服务器的框架下，延迟的步数也可以由用户自己决定，下图分别刻划了几种延迟情况：

![img](./assets/readme/v2-d6c9e37470f7129d76cae642038c8e0a_1440w.jpg)

- **(a) 无延迟**
- **(b) 延迟但不指定延迟步数**。也即在迭代2时，用的可能是老权重，也可能是新权重，听天由命。
- **(c) 延迟且指定延迟步数为1**。例如做迭代3时，可以不拿回迭代2的梯度，但必须保证迭代0、1的梯度都已拿回且用于参数更新。

总结，**异步很香，但对一个Worker来说，只是等于W不变，batch的数量增加了而已，在SGD下，会减慢模型的整体收敛速度**。`异步的整体思想是，比起让Worker闲着，倒不如让它多吃点数据，虽然反馈延迟了，但只要它在干活在学习就行。`

参考：https://zhuanlan.zhihu.com/p/617133971





#### DDP

DDP 过程

![ddp](./assets/readme/ddp.png)

 ![ddp2](./assets/readme/ddp2.png)

 

不再有主GPU，每个GPU执行相同的任务。对每个GPU的训练都是在自己的进程中进行的。每个进程都从磁盘加载其自己的数据。分布式数据采样器可确保加载的数据在各个进程之间不重叠。

损失函数的前向传播和计算在每个GPU上独立执行。因此，不需要收集网络输出。在反向传播期间，梯度下降在所有GPU上均被执行，从而确保每个GPU在反向传播结束时最终得到平均梯度的相同副本。

与DataParallel的单进程控制多GPU不同，在distributed的帮助下，只需要编写一份代码，torch就会自动将其分配给n个进程，分别在 n 个GPU上运行。

```python3

#详细步骤如下：
#1.导入必要的模块：
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

#2.初始化torch.distributed包：
dist.init_process_group(backend='nccl')
#设置通信后端为NCCL，并协调设备或机器之间的进程组形成。

#3.定义模型并将其包装在DistributedDataParallel中：
model = Model()
model = DistributedDataParallel(model)
#将模型包装在DDP中，以在多个设备或机器上进行并行训练。

#4.将数据加载到每个进程中：
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

#5.调整每个进程的批量大小：
batch_size_per_process = int(batch_size / dist.get_world_size())
#确保每个进程都有足够的数据进行处理。

#6.将模型和输入张量移动到正确的设备上：
model = model.to(device)
input = input.to(device)

#7.在训练循环中，执行前向传播、反向传播和优化器更新：
output = model(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()

#8.在训练结束后，将模型的权重和偏置参数同步到所有设备或机器：
model_state_dict = model.module.state_dict()
dist.broadcast(model_state_dict, src=0)
model.module.load_state_dict(model_state_dict)
#确保所有设备上的模型参数保持一致。

#9.清理分布式环境：
dist.destroy_process_group()
```

### 模型并行

#### Tensor并行

![image-20231225104104256](assets/readme/image-20231225104104256.png)

在上面的这张图里，每一个节点（或者叫进程）都有一份模型，然后各个节点取不同的数据，通常是一个batch_size，然后各自完成前向和后向的计算得到梯度，这些进行训练的进程我们成为worker，除了worker，还有参数服务器，简称ps server，这些worker会把各自计算得到的梯度送到ps server，然后由ps server来进行update操作，然后把update后的模型再传回各个节点。因为在这种并行模式中，被划分的是数据，所以这种并行方式叫数据并行

![image-20231225104154125](assets/readme/image-20231225104154125.png)

深度学习的计算其实主要是矩阵运算，而在计算时这些矩阵都是保存在内存里的，如果是用GPU卡计算的话就是放在显存里，可是有的时候矩阵会非常大，比如在CNN中如果num_classes达到千万级别，那一个FC层用到的矩阵就可能会大到显存塞不下。这个时候就不得不把这样的超大矩阵给拆了分别放到不同的卡上去做计算，从网络的角度来说就是把网络结构拆了，其实从计算的过程来说就是把矩阵做了分块处理。

比较好理解，具体看Megatron论文，就是把一个神经网络层Tensor切成了多个小的Tensor，每个tensor放在不同的gpu。主要就是列并行、行并行。在transformer里的应用具体体现在MLP、Attention层里。

![image-20231225110603413](assets/readme/image-20231225110603413.png)



有的时候呢数据并行和模型并行会被同时用上。比如深度的卷积神经网络中卷积层计算量大，但所需参数系数 W 少，而FC层计算量小，所需参数系数 W 多。因此对于卷积层适合使用数据并行，对于全连接层适合使用模型并行。 就像这样

![image-20231225104358084](assets/readme/image-20231225104358084.png)

#### 流水线并行

流水并行是指按顺序将模型切分为不同的部分至不同的GPU上运行。每个GPU上只有部分参数，因此每个部分的模型消耗GPU的显存成比例减少。

将大型模型分为若干份连续的layer很简单。但是，layer的输入和输出之间存在顺序依赖关系，因此在一个GPU等待其前一个GPU的输出作为其输入时，朴素的实现会导致出现大量空闲时间。这些空闲时间被称作“气泡”，而在这些等待的过程中，空闲的机器本可以继续进行计算。

![image-20231225110835198](assets/readme/image-20231225110835198.png)

一个朴素的流水并行设置，其中模型按layer垂直分成 4 个部分。worker 1托管网络第一层（离输入最近）的模型参数，而 worker 4 托管第 4 层（离输出最近）的模型参数。“F”、“B”和“U”分别代表前向、反向和更新操作。下标指示数据在哪个节点上运行。由于顺序依赖性，数据一次只能在一个节点上运行。

了减少气泡的开销，在这里可以复用数据并行的打法，核心思想是将大批次数据分为若干个微批次数据（microbatches），每个节点每次只处理一个微批次数据，这样在原先等待的时间里可以进行新的计算。


每个微批次数据的处理速度会成比例地加快，每个节点在下一个小批次数据释放后就可以开始工作，从而加快流水执行。有了足够的微批次，节点大部分时间都在工作，而气泡在进程的开头和结束的时候最少。梯度是微批次数据梯度的平均值，并且只有在所有小批次完成后才会更新参数。

模型拆分的节点数通常被称为流水线深度（pipeline depth）。

在前向传递过程中，节点只需将其layer块的输出（激活）发送给下一个节点；在反向传递过程中，节点将这些激活的梯度发送给前一个节点。如何安排这些进程以及如何聚合微批次的梯度有很大的设计空间。GPipe 让每个节点连续前向和后向传递，在最后同步聚合多个微批次的梯度。PipeDream则是让每个节点交替进行前向和后向传递。

![image-20231225110938859](assets/readme/image-20231225110938859.png)

GPipe 和 PipeDream 流水方案对比。每批数据分为4个微批次，微批次1-8对应于两个连续大批次数据。图中，“（编号）”表示在哪个微批次上执行操作，下标表示节点 ID。其中，PipeDream使用相同的参数执行计算，可以获得更高的效率。



**参考：**

- https://www.zhihu.com/question/53851014
- [流水并行讲的好](https://www.zhihu.com/question/53851014/answer/2530594788)
- [图解大模型训练之：数据并行上篇(DP, DDP与ZeRO)](https://zhuanlan.zhihu.com/p/617133971)
- [图解大模型训练之：数据并行下篇( DeepSpeed ZeRO，零冗余优化)](https://zhuanlan.zhihu.com/p/618865052)

## ZeRO

目前训练超大规模语言模型技术路线：GPU + PyTorch + Megatron-LM + DeepSpeed。

 **ZeRO介绍**

在使用 ZeRO 进行分布式训练时，可以选择 ZeRO-Offload 和 ZeRO-Stage3 等不同的优化技术。严格来讲ZeRO采用数据并行+张量并行的方式，旨在降低存储。

ZeRO-Offload和ZeRO-Stage3是DeepSpeed中的不同的Zero-Redundancy Optimization技术，用于加速分布式训练，主要区别在资源占用和通信开销方面。

ZeRO-Offload将模型参数分片到不同的GPU上，通过交换节点间通信来降低显存占用，但需要进行额外的通信操作，因此可能会导致训练速度的下降。

ZeRO-Stage3将模型参数分布在CPU和GPU上，通过CPU去计算一部分梯度，从而减少显存占用，但也会带来一定的计算开销。

**ZeRO的三个级别**：

ZeRO-0：禁用所有类型的分片，仅使用 DeepSpeed 作为 DDP (Distributed Data Parallel)

ZeRO-1：分割Optimizer States，减少了4倍的内存，通信容量与数据并行性相同

ZeRO-2：分割Optimizer States与Gradients，8x内存减少，通信容量与数据并行性相同

ZeRO-3：分割Optimizer States、Gradients与Parameters，内存减少与数据并行度和复杂度成线性关系。

ZeRO-Infinity是ZeRO-3的拓展。允许通过使用 NVMe 固态硬盘扩展 GPU 和 CPU 内存来训练大型模型。ZeRO-Infinity 需要启用 ZeRO-3。

ZeRO-DP可以分为三个阶段：Pos, Pg, Pp 。三个阶段对应优化器状态划分、梯度划分和模型参数划分，并且三个阶段可以叠加使用(上图展示了三个阶段的叠加)。ZeRO的几个级别如下图2所示，假设模型参数量为7.5B，不同级别的显存情况如下。

![ZeRO1](./assets/readme/ZeRO1.png)

上图是ZeRO不同级别内存划分

![ZeRO2](./assets/readme/ZeRO2.png)

上图是不同参数量的切分

## DeepEP

DeepEP 是一个针对GPU硬件专门为 MoE 专家并行（EP）定制的通信库。它为 AllToAll 全连接通信算子提供高吞吐量和低延迟的通信接口，使用GPU内核加速实现。支持低精度操作，包括 FP8。鉴于对 AI 中高效分布式计算需求的不断增长，DeepEP 在提升性能方面的作用至关重要，尤其是在实时应用和大规模模型训练方面。

### 专家并行的通信需求

MoE 将 AI 模型划分为多个专门的子网络，或称为“专家”，每个专家在不同的数据子集上训练。一个门控网络决定哪个专家处理给定的输入，从而提高效率和准确性。例如，混合专家解释指出，与密集模型相比，MoE 模型能够实现更快的预训练，降低计算成本。这种方法对于大型语言模型尤其有益，允许在不成比例增加计算量的情况下实现可扩展性。

专家并行是一种模型并行形式，其中这些专家子网络分布在不同的计算设备上，例如 GPU。这种分布减少了每个设备的内存需求，同时保持计算效率。

![MoE](./assets/readme/MoE.png)

如图所示，不同的token的输入经过router门控算法后的结果需要发到不同的expert上运行，专家并行就是需要不同的expert算法运行在不同的GPU上。各个GPU之间的通信其是一个AllToAll通信，由于NCCL通信库是没有实现AllToAll通信接口的，需要使用send 和 recv接口实现。

### DeepEP的主要特性

①节点内高吞吐量通信：

使用 NVLink 进行节点内全对全通信，适用于高带宽需求的场景。

②节点间高吞吐量通信：

使用 RDMA（Remote Direct Memory Access）进行节点间全对全通信。使用 RDMA（远程直接内存访问）进行节点间全对全通信。支持高吞吐量场景，适用于大规模分布式训练。

③混合场景通信：

在同一节点内的通信使用 NVLink，跨节点通信使用 RDMA。通过 Buffer 类的 internode_dispatch 和 internode_combine 方法实现。

④低延迟通信：

使用 RDMA 的 IBGDA（InfiniBand 全局设备访问）功能，支持低延迟的全对全通信。

IBGDA允许 GPU 直接访问远程 GPU 的内存，避免数据包在 NVLink 和 RDMA 之间的转发。利用 NVSHMEM 的 GPU 直接 RDMA 特性，绕过 CPU，实现 GPU 内存到远程 GPU 内存的直接传输。NVSHMEM是NVIDIA推出的并行编程接口，专为GPU集群设计，通过对称内存模型实现高效跨节点通信。其核心功能包括支持多节点、多互连（如NVLink/InfiniBand）的通信架构。

⑤支持 FP8：

启用低精度操作以减少内存使用并提高性能，与现代硬件能力相匹配。

⑥基于钩子的通信-计算重叠：

允许在不占用流多处理器（SMs）的情况下重叠通信和计算，提高资源利用率。

### DeepEP对外接口

整个代码对外提供的接口包括：

节点内通信接口为 `intranode_dispatch、intranode_combine`

节点间通信接口为 `internode_dispatch、internode_combine`

低延时接口包括 `low_latency_dispatch、low_latency_combine`

DeepEP 提供两种类型的内核，每种内核都针对特定的用例进行了优化。

***普通内核：***

关注高吞吐量，适用于训练和推理预填充。

支持 NVLink 和 RDMA 通信，确保节点内和节点间数据传输高效。

针对大批量数据优化，充分利用 GPU 资源。

***低延迟内核：***

基于纯 RDMA，最小化实时推理解码的延迟。

适用于小批量数据，对延迟敏感的应用至关重要。

包含基于钩子的重叠方法，通信和计算之间实现并行处理，确保不占用 SM 资源。

**通信不占用 SMs 的关键在于利用 DMA 引擎。GPU 的 DMA 引擎可以独立于 SMs 进行内存传输操作，因此通信任务可以在后台进行，而 SMs 继续执行计算任务。通过 CUDA 流管理实现，将通信任务分配到与计算任务不同的流上，通过事件同步确保正确顺序。**
