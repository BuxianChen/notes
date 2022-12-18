
# 学习路线及资料

- Compiler:
  - 编译原理: B站华保健老师课程及作业
  - LLVM: GiantPandasCV B站视频
  - TVM: 陈天奇B站视频及讲义
  - pytorch: pytorch 2.0, torch.fx, torch.jit.trace, torch.jit.script
- Inference Engine:
  - mmdeploy(优先级高): 官方文档
  - onnx(优先级高): 官方文档
  - tensorRT:
- 追剧计划:
  - GiantPandasCV
  - mmlab
  - 李沐
- 其他前置知识
  - cmake, make(优先级高)
  - pybind11(优先级高)


各个知识点之间的联系:

部署流程:

最终一般是用推理框架来完成的, 例如: tensorRT, ncnn, onnxruntime。这些框架都各自定义了一套模型存储格式, 根据这套存储格式适配一种或多种硬件做推理。而当前主流的训练框架有pytorch, tensorflow等, 他们各自定义了一套模型存储格式, 因此本质上需要:

```
torch2tensorRT, torch2ncnn, torch2onnxruntime, tf2tensorRT, tf2ncnn, tf2onnxruntime
```

因此可能需要 $m \times n$ 种转换, 因此 onnx 作为中间格式应运而生, 这样子只需要 $m+n$ 种转换:

```
# 备注: 这里onnx2onnxruntime实际上是不需要的
torch2onnx, tf2onnx, onnx2tensorRT, onnx2ncnn, onnx2onnxruntime
```

注意: 可以看出, 实际上部署流程并非都得走 torch(.pth) -> onnx(.onnx) -> tensorRT(.engine) 这种流程, 能直接转成(.engine)格式实际就可以达到目的, 只是因为"生态"的原因, 通常会走两步转换的流程.


## make, cmake

```
C_INCLUDE_PATH  # C 头文件库搜索路径, 备注: 系统本身的不在这个变量里
CPLUS_INCLUDE_PATH  # C++ 头文件库搜索路径, 备注: 系统本身的不在这个变量里
LD_LIBRARY_PATH  # 动态链接库搜索路径, 备注: 系统本身的不在这个变量里
LIBRARY_PATH  # # 静态链接库搜索路径, 备注: 系统本身的不在这个变量里
```

为什么通常会需要配置 `LD_LIBRARY_PATH` 而不需要配置 `C_INCLUDE_PATH` 和 `CPLUS_INCLUDE_PATH`, 例如:

- 安装 CUDA
- 安装 openCV
- 安装 tensorRT

### gcc/g++ 与 llvm

?


## pybind11

### 相关的东西

结论: 目前最为流行的方式是 pybind11, 许多开源项目一般将 pybind11 作为 git submodule 放在 `third_party` 文件夹中, 源码编译这些开源项目会用到 pybind11, 例如:

- onnx: https://github.com/onnx/onnx/tree/main/third_party
- pytorch: https://github.com/pytorch/pytorch/tree/master/third_party
- tvm: 不确定是否使用 pybind11
- tensorflow: 不确定是否使用 pybind11
- mmdeploy: https://github.com/open-mmlab/mmdeploy/tree/master/third_party
- mxnet: 不确定是否使用 pybind11
- faiss: 似乎不是 pybind11, https://github.com/facebookresearch/faiss
- SPTAG: 使用 SWIG, https://github.com/microsoft/SPTAG

#### cpython 的原生方式

cpython 原生的 C 拓展的方式为:

pybind11实际上是对这种拓展方式做了层层封装

#### ctypes

#### cython, swig

略



## onnx

### Protocol Buffer

Google定义了一套用于代替xml的文本文件格式, 并提供了一套完整的库来解析, 序列化这种数据格式, onnx的序列化使用了这种格式

官方文档: https://developers.google.com/protocol-buffers/docs

### onnx 概念

### onnx Python API



