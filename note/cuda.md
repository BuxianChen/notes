
# 参考资料

- [https://shiyan.medium.com/some-cuda-concepts-explained-12ecc390d10f](https://shiyan.medium.com/some-cuda-concepts-explained-12ecc390d10f)

# 基本概念

术语方面, GPU 硬件与 CUDA 软件术语有些使用了同样的词。

一个 GPU 由如下部分组成 (参考 [https://medium.com/analytics-vidhya/cuda-memory-model-823f02cef0bf](https://medium.com/analytics-vidhya/cuda-memory-model-823f02cef0bf))

- 多个 SM (Streaming Multiprocessor, 有时候也被简称为 Multiprocessor)
    - 多个 CUDA Cores
    - Shared Memory
    - L1 Cache
    - Read-Only DataCache
- L2 Cache
- DRAM (GPU主存)

使用 [cuda-samples](https://github.com/NVIDIA/cuda-samples.git) 可以查看 GPU 的设备信心

```bash
git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples
git checkout v11.2  # 切换到 nvcc -V 的版本
cd /Samples/deviceQuery
make
./deviceQuery
```

输出 (重新 format 了一下, 便于对比, 这里 GeForce MX250 是 Windows 本机搭配 WSL2, Tesla V100-PCIE-16GB 是一个虚拟化的云主机):

```
                                                 NVIDIA GeForce MX250                                       Tesla V100-PCIE-16GB
  CUDA Driver Version / Runtime Version          12.2 / 11.2                                                12.2 / 11.2
  CUDA Capability Major/Minor version number:    6.1                                                        7.0
  Total amount of global memory:                 2048 MBytes (2147352576 bytes)                             16151 MBytes (16935419905 bytes)
  ( 3) Multiprocessors, (128) CUDA Cores/MP:     384 CUDA Cores                                             (080) Multiprocessors, (064) CUDA Cores/MP: 5120 CUDA Cores
  GPU Max Clock rate:                            1038 MHz (1.04 GHz)                                        1380 MHz (1.38 GHz)
  Memory Clock rate:                             3004 Mhz                                                   877 MHz
  Memory Bus Width:                              64-bit                                                     4096-bit
  L2 Cache Size:                                 524288 bytes                                               6291456 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)  1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers                                    1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers                             2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes                                                65536 bytes
  Total amount of shared memory per block:       49152 bytes                                                49152 bytes
  Total shared memory per multiprocessor:        98304 bytes                                                98304 bytes
  Total number of registers available per block: 65536                                                      65536
  Warp size:                                     32                                                         32
  Maximum number of threads per multiprocessor:  2048                                                       2048
  Maximum number of threads per block:           1024                                                       1024
  Max dimension size of a thread block (x,y,z):  (1024, 1024, 64)                                           (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z):  (2147483647, 65535, 65535)                                 (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes                                           2147483647 bytes
  Texture alignment:                             512 bytes                                                  512 bytes
  Concurrent copy and kernel execution:          Yes with 1 copy engine(s)                                  Yes with 7 copy engine(s)
  Run time limit on kernels:                     Yes                                                        No
  Integrated GPU sharing Host Memory:            No                                                         No
  Support host page-locked memory mapping:       Yes                                                        Yes
  Alignment requirement for Surfaces:            Yes                                                        Yes
  Device has ECC support:                        Disabled                                                   Enabled
  Device supports Unified Addressing (UVA):      Yes                                                        Yes
  Device supports Managed Memory:                Yes                                                        Yes
  Device supports Compute Preemption:            Yes                                                        Yes
  Supports Cooperative Kernel Launch:            Yes                                                        Yes
  Supports MultiDevice Co-op Kernel Launch:      No                                                         Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0                                                  0 / 177 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >
```

这里我们先关注这几个值 (以 V100 为例):

```
// int dev = 0;
// cudaSetDevice(dev);
// cudaDeviceProp deviceProp;
// cudaGetDeviceProperties(&deviceProp, dev);

(080) Multiprocessors, (064) CUDA Cores/MP:    5120 CUDA Cores  // deviceProp.multiProcessorCount, _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor)
Maximum number of threads per multiprocessor:  2048             // deviceProp.maxThreadsPerMultiProcessor
Maximum number of threads per block:           1024             // deviceProp.maxThreadsPerBlock
Max dimension size of a thread block (x,y,z):  (1024, 1024, 64) // deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]
Max dimension size of a grid size (x,y,z):     (2147483647, 65535, 65535)  // deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]
```

- `deviceProp.maxThreadsPerBlock=1024`: CUDA 编程模型视角里 block 的三个维度之积不能超过 1024
- `deviceProp.maxThreadsPerMultiProcessor=2048`: 一个 SM 可以同时执行的最大硬件线程数为 2048, 注意从 CUDA 编程视角来看, 一个 block 里的所有线程都会被运行在同一个 SM 上 (注意可能不会是并发执行的), 而最终在运行 thread 时, thread 会映射到硬件线程上, 例如 block 的三个维度之积为 68 (这不是最佳实践, 假设硬件的 wrap 的大小为 32), 那么这 68 个软件意义上的线程会被分为 32+32+4 三组, 其中每组内的软件线程总是会映射到同一个硬件线程束 (wrap) 上并发执行的, 但这三组 wrap 有可能不是并发执行的, 但无论如何, 这 68 个线程一定都在同一个 SM 上执行.
- `deviceProp.multiProcessorCount=80`: GPU 包含 80 个 SM
- `_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor)=64`: 每个 SM 包含 64 个 CUDA Core

- `deviceProp.maxThreadsDim=(1024, 1024, 64)`. 这代表了 CUDA 编程模型视角里每个 block 的软件线程数在三个维度上的最大值, 注意还需满足三个维度之积不超过 `deviceProp.maxThreadsPerBlock=1024`
- `deviceProp.maxGridSize=(2147483647, 65535, 65535)`. 这代表了 CUDA 编程模型视角里每个 grid 的 block 数在三个维度上的最大值, 也就是说一个核函数最多只能由 `2147483647*65535*65535*1024` 个线程来完成整个任务.

关于 wrap: wrap 是硬件层的概念, 一个 wrap 里的 32 个物理线程严格并发执行, 且并发执行的指令一模一样, 当然, 操作数可以是不一样的 (这种模式也被称作 SIMD, 即 Single Instruction, Multiple Data).

关于 block: block 是纯粹的软件视角的概念

wrap 与 CUDA Core 的关系: 这篇 [博客](https://shiyan.medium.com/some-cuda-concepts-explained-12ecc390d10f) 里有个误解是 1 个 CUDA Core 就对应 1 个 wrap, 但根据这个[问答](https://stackoverflow.com/questions/16986770/cuda-cores-vs-thread-count/16987220#16987220):

> Now your Card has a total Number of 384 cores on 2 SMs with 192 cores each. The CUDA core count represents the total number of single precision floating point or integer thread instructions that can be executed per cycle. Do not consider CUDA cores in any calculation.

我们在使用这种方式调用核函数时:

```c++
int a, b, c;  // 每个 block 的 thread 数为 a*b*c
int x, y, z;  // 调用整个核函数完成整个功能需要 x*y*z 个 block
dim3 threads(a, b, c);
dim3 grid(x, y, z);
kernel_fun <<<grid, threads>>> ();
```

理论上说:

```c++
// (a, b, c) <= (1024, 1024, 64), a*b*c <= 1024
// (x, y, z) <= (2147483647, 65535, 65535)
a=64, b=64, c=2                     // 非法, a*b*c 超过了 deviceProp.maxThreadsPerBlock 的限制
a=1024, b=1, c=1, x=160, y=1, z=1   // 理论上可以利用完整个 GPU 的并发: 每个 SM 同时运行 2 个 block
a=32, b=4, c=4, x=320, y=1, z=1     // 理论上可以利用完整个 GPU 的并发: 每个 SM 同时运行 4 个 block
a=68, b=1, c=1, x=320, y=320, z=320 // a*b*c=68 不是 32 的倍数, 强烈不推荐
a=1024, b=1, c=1, x=320, y=1, z=1   // 理论上可以利用完整个 GPU 的并发, 只是需要 2 次完全并发才能完成整个任务
```


FAQ:

compute-capability 与 cuda-architecture 是同一个意思: [问答](https://stackoverflow.com/questions/65097396/difference-between-compute-capability-cuda-architecture-clarification-for-us)
