
# 参考资料

官方资料:

- [https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)

其他

- 英伟达各型号 GPU 的关键参数: [https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units](https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units)
- [https://shiyan.medium.com/some-cuda-concepts-explained-12ecc390d10f](https://shiyan.medium.com/some-cuda-concepts-explained-12ecc390d10f)

一些未仔细核验的资料:

- 一个异构编程的课程: [https://safari.ethz.ch/projects_and_seminars/spring2022/doku.php?id=heterogeneous_systems](https://safari.ethz.ch/projects_and_seminars/spring2022/doku.php?id=heterogeneous_systems)


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

机器信息:

- GeForce MX250: Windows + WSL2
- Tesla V100-PCIE-16GB: 虚拟化 linux 云主机
- GeForce GTX 1650: Windows + WSL2

使用 [cuda-samples](https://github.com/NVIDIA/cuda-samples.git) 可以查看 GPU 的设备信心

```bash
git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples
git checkout v11.2  # 切换到 nvcc -V 的版本
cd /Samples/deviceQuery
make
./deviceQuery
```

输出 (重新 format 了一下, 便于对比):

```
                                                 NVIDIA GeForce MX250                                       Tesla V100-PCIE-16GB                                          NVIDIA GeForce GTX 1650
  CUDA Driver Version / Runtime Version          12.2 / 11.2                                                12.2 / 11.5                                                   12.2 / 11.2
  CUDA Capability Major/Minor version number:    6.1                                                        7.0                                                           7.5
  Total amount of global memory:                 2048 MBytes (2147352576 bytes)                             16151 MBytes (16935419905 bytes)                              4096 MBytes (4294639616 bytes)
  ( 3) Multiprocessors, (128) CUDA Cores/MP:     384 CUDA Cores                                             (080) Multiprocessors, (064) CUDA Cores/MP: 5120 CUDA Cores   (14) Multiprocessors, (64) CUDA Cores/MP: 896 CUDA Cores
  GPU Max Clock rate:                            1038 MHz (1.04 GHz)                                        1380 MHz (1.38 GHz)                                           1710 MHz (1.71 GHz)
  Memory Clock rate:                             3004 Mhz                                                   877 MHz                                                       6001 Mhz
  Memory Bus Width:                              64-bit                                                     4096-bit                                                      128-bit
  L2 Cache Size:                                 524288 bytes                                               6291456 bytes                                                 1048576 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)  1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)     1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers                                    1D=(32768), 2048 layers                                       1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers                             2D=(32768, 32768), 2048 layers                                2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes                                                65536 bytes                                                   65536 bytes
  Total amount of shared memory per block:       49152 bytes                                                49152 bytes                                                   49152 bytes
  Total shared memory per multiprocessor:        98304 bytes                                                98304 bytes                                                   65536 bytes
  Total number of registers available per block: 65536                                                      65536                                                         65536
  Warp size:                                     32                                                         32                                                            32
  Maximum number of threads per multiprocessor:  2048                                                       2048                                                          1024
  Maximum number of threads per block:           1024                                                       1024                                                          1024
  Max dimension size of a thread block (x,y,z):  (1024, 1024, 64)                                           (1024, 1024, 64)                                              (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z):  (2147483647, 65535, 65535)                                 (2147483647, 65535, 65535)                                    (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes                                           2147483647 bytes                                              2147483647 bytes
  Texture alignment:                             512 bytes                                                  512 bytes                                                     512 bytes
  Concurrent copy and kernel execution:          Yes with 1 copy engine(s)                                  Yes with 7 copy engine(s)                                     Yes with 6 copy engine(s)
  Run time limit on kernels:                     Yes                                                        No                                                            Yes
  Integrated GPU sharing Host Memory:            No                                                         No                                                            No
  Support host page-locked memory mapping:       Yes                                                        Yes                                                           Yes
  Alignment requirement for Surfaces:            Yes                                                        Yes                                                           Yes
  Device has ECC support:                        Disabled                                                   Enabled                                                       Disabled
  Device supports Unified Addressing (UVA):      Yes                                                        Yes                                                           Yes
  Device supports Managed Memory:                Yes                                                        Yes                                                           Yes
  Device supports Compute Preemption:            Yes                                                        Yes                                                           Yes
  Supports Cooperative Kernel Launch:            Yes                                                        Yes                                                           Yes
  Supports MultiDevice Co-op Kernel Launch:      No                                                         Yes                                                           No
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0                                                  0 / 177 / 0                                                   0 / 1 / 0
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
- `deviceProp.maxThreadsPerMultiProcessor=2048`: 一个 SM 可以同时执行的最大硬件线程数为 2048, 注意从 CUDA 编程视角来看, 一个 block 里的所有线程都会被运行在同一个 SM 上 (注意可能不会是并发执行的), 而最终在运行 thread 时, thread 会映射到硬件线程上, 例如 block 的三个维度之积为 68 (这不是最佳实践, 假设硬件的 warp 的大小为 32), 那么这 68 个软件意义上的线程会被分为 32+32+4 三组, 其中每组内的软件线程总是会映射到同一个硬件线程束 (warp) 上并发执行的, 但这三组 warp 有可能不是并发执行的, 但无论如何, 这 68 个线程一定都在同一个 SM 上执行.
- `deviceProp.multiProcessorCount=80`: GPU 包含 80 个 SM
- `_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor)=64`: 每个 SM 包含 64 个 CUDA Core

- `deviceProp.maxThreadsDim=(1024, 1024, 64)`. 这代表了 CUDA 编程模型视角里每个 block 的软件线程数在三个维度上的最大值, 注意还需满足三个维度之积不超过 `deviceProp.maxThreadsPerBlock=1024`
- `deviceProp.maxGridSize=(2147483647, 65535, 65535)`. 这代表了 CUDA 编程模型视角里每个 grid 的 block 数在三个维度上的最大值, 也就是说一个核函数最多只能由 `2147483647*65535*65535*1024` 个线程来完成整个任务.

~~关于 warp: warp 是硬件层的概念, 一个 warp 里的 32 个物理线程严格并发执行, 且并发执行的指令一模一样, 当然, 操作数可以是不一样的 (这种模式也被称作 SIMD, 即 Single Instruction, Multiple Data).~~

关于 block: block 是纯粹的软件视角的概念

warp 与 CUDA Core 的关系: 这篇 [博客](https://shiyan.medium.com/some-cuda-concepts-explained-12ecc390d10f) 里有个误解是 1 个 CUDA Core 就对应 1 个 warp, 但根据这个[问答](https://stackoverflow.com/questions/16986770/cuda-cores-vs-thread-count/16987220#16987220):

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

## threadIdx, blockDim, blockIdx, gridDim

[https://erangad.medium.com/1d-2d-and-3d-thread-allocation-for-loops-in-cuda-e0f908537a52](https://erangad.medium.com/1d-2d-and-3d-thread-allocation-for-loops-in-cuda-e0f908537a52)

```c++
__global__ void kernel_fun(){
  threadIdx.x < blockDim.x;
  threadIdx.y < blockDim.y;
  threadIdx.z < blockDim.z;
  (blockDim.x == 2) && (blockDim.y==16) && (blockDim.z==4)

  blockIdx.x < gridDim.x;
  blockIdx.y < gridDim.y;
  blockIdx.z < gridDim.z;
  (gridDim.x == 2) && (gridDim.y==3) && (gridDim.z==6)
}

int a=2, b=16, c=4;
int x=2, y=3, z=6;

dim3 block_dim(a, b, c);
dim3 thread_dim(x, y, z);

kernel_fun <<<grid, threads>>> ();
```

**注意: `threadIdx.x` 和 `blockIdx.x` 是变化最快的维度**: 上面的例子里: `threadIdx.x=0, threadIdx.y=1, threadIdx.z=2` 的下一个 thread 是 `threadIdx.x=1, threadIdx.y=1, threadIdx.z=2`, 即:

```
// Thread 0 - Thread 31 会组成一个 warp.
Thread 0: (threadIdx.x, threadIdx.y, threadIdx.z) = (0, 0, 0)
Thread 1: (threadIdx.x, threadIdx.y, threadIdx.z) = (1, 0, 0)
Thread 2: (threadIdx.x, threadIdx.y, threadIdx.z) = (2, 0, 0)
Thread 3: (threadIdx.x, threadIdx.y, threadIdx.z) = (3, 0, 0)
Thread 4: (threadIdx.x, threadIdx.y, threadIdx.z) = (0, 1, 0)
Thread 5: (threadIdx.x, threadIdx.y, threadIdx.z) = (1, 1, 0)
```

## 执行逻辑

参考: [https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-5-x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-5-x)

假设一次调用包含 8 个 block, 每个 block 中有 512 个 thread, 即使用这种方式进行调用, `kernel_fun <<<8, 512>>> (p);` 假设 warp 大小为 32, 那么每个 block 的前 32 个 thread 将会最终映射到一个 warp 上, 后 32 个也会映射到一个 warp 上 (warp 不是硬件概念, 而是意味着前 32 个 thread 在同一时钟周期内会执行相同的指令). 假设 GPU 有 80 个 SM, 而每个 SM 包含 4 个 warp scheduler, 那么当一个 block 被调度到一个 SM 上后 (block 一旦被调度到 SM, 那么一定会将其完成, 不会被切换到别的 SM 上), SM 会进一步分配给 4 个 warp scheduler 来处理, 在前面的例子里, 一个 block 被分为 16 组, 一个可能的情况是:

```
0号调度器：负责执行 0，4，7，9，10，14 组
1号调度器：负责执行 1，5，8，11 组
2号调度器：负责执行 2，3，6 组
3号调度器：负责执行 12，13，15 组
```

而以 2 号调度器为例, 假设之前描述的每个 thread 只包含 2 个指令, 具体的执行顺序可能是:

```
block3-warp2-instruct0
block3-warp6-instruct0
block3-warp2-instruct1
block3-warp3-instruct0
block3-warp3-instruct1
block3-warp6-instruct1
```

注意, 在 warp2, warp3, warp6 在执行过程中, 有可能会插入别的 block 的执行, 但是 warp scheduler 的同一时间, 只能执行一个 warp. 也就是说在这个例子里, 一个 SM 的最大并发量是 `4 * 32 = 128` 个线程.


## FAQ & 杂录

compute-capability 与 cuda-architecture 是同一个意思: [问答](https://stackoverflow.com/questions/65097396/difference-between-compute-capability-cuda-architecture-clarification-for-us)

最大单精度浮点数计算次数: MX250: 797.2 GFLOPS, V100: 14028 GFLOPS, GTX1650: 2984 GFLOPS

# 矩阵乘法

benchmark for (1024 x 1024) x (1024 x 1024)

- cuda-sample/matrixMul
  - MX250: 140 GFLOPS / 797.2 GFLOPS
  - V100: 3800 GFLOPS / 14028 GFLOPS
  - GTX1650: 426 GFLOPS / 2984 GFLOPS
