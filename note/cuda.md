
# 硬件

术语方面, GPU 硬件与 CUDA 软件术语有些使用了同样的词【例如】, 本节所有的术语都是硬件层面的术语, 之后的节引用硬件层面的术语都加上 "硬件-" 前缀。

[https://medium.com/analytics-vidhya/cuda-memory-model-823f02cef0bf](https://medium.com/analytics-vidhya/cuda-memory-model-823f02cef0bf)

一个 GPU 由如下部分组成

- 多个SM (Streaming Multiprocessor)
    - 多个 CUDA Cores
    - Shared Memory
    - L1 Cache
    - Read-Only DataCache
- L2 Cache
- DRAM (GPU主存)

compute-capability 与 cuda-architecture 是同一个意思: [问答](https://stackoverflow.com/questions/65097396/difference-between-compute-capability-cuda-architecture-clarification-for-us)

# 软件

同一个 block 中的 thread 都会被调度到同一个 SM 上执行, 且执行相对顺序固定为:

- 第一组 32 个线程并发执行
- 第二组 32 个线程并发执行