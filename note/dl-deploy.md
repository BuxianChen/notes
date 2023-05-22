
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


## make, cmake, gcc/g++, llvm

参见[个人笔记](../tool/make.md)


## pybind11

结论: 目前最为流行的方式是 pybind11, 许多开源项目一般将 pybind11 作为 git submodule 放在 `third_party` 文件夹中, 源码编译这些开源项目会用到 pybind11, 例如:

- onnx: https://github.com/onnx/onnx/tree/main/third_party
- pytorch: https://github.com/pytorch/pytorch/tree/master/third_party
- tvm: 不确定是否使用 pybind11
- tensorflow: 不确定是否使用 pybind11
- mmdeploy: https://github.com/open-mmlab/mmdeploy/tree/master/third_party
- mxnet: 不确定是否使用 pybind11
- faiss: 似乎不是 pybind11, https://github.com/facebookresearch/faiss
- SPTAG: 使用 SWIG, https://github.com/microsoft/SPTAG


cpython 原生的 C 拓展的方式为:

pybind11实际上是对这种拓展方式做了层层封装

存在其他的python调用C/C++扩展的方式:
- ctypes
- cython
- swig

详细内容参见 [个人博客](https://buxianchen.github.io/drafts/2022-06-25-pybind.html)


## onnx

### Protocol Buffer (Finished)

Google定义了一套用于代替xml,json的格式, 并提供了一套完整的库来解析, 序列化这种数据格式, onnx的序列化使用了这种格式

官方文档: https://developers.google.com/protocol-buffers/docs

安装步骤参考官方教程进行, 此处简单记一下，两项均需要安装:
- 安装 protoc（任选其一）:
  - 从官方 Github 的 Release 中下载关于 protoc 的压缩包，里面有预编译好的二进制 protoc 文件，并将它添加至 `$PATH` 变量中
  - 下载源码按官方教程编译出 protoc
- 安装 protobuf runtime (此处仅记录python，以下方式任选其一)
  - pip install protobuf==xx.xx.xx
  - 从官方 Github 的 Release 中下载源代码, 进入`python` 文件夹后, 执行 `python setup build` 与 `python setup install`


python 使用 protobuf 分为如下几步:
- 定义数据规范: 即声明字段及字段的类型等(即定义结构体), 这种声明使用的是一种特殊的语法, 保存在一个 `.proto` 文件中
- 使用 `protoc` 命令工具用 `.proto` 文件生成一个 `.py` 文件(自动生成代码)
- import 这个生成的文件以创建 `.proto` 里声明的数据结构对象, 并可以使用相关方法将数据解析或保存到文件中

示例（来源于[官方教程](https://developers.google.com/protocol-buffers/docs/pythontutorial)）:

**第一步: 定义"数据结构"**

```protobuf
// addressbook.proto
syntax = "proto2";

package tutorial;

message Person {
  optional string name = 1;
  optional int32 id = 2;
  optional string email = 3;

  enum PhoneType {
    MOBILE = 0;
    HOME = 1;
    WORK = 2;
  }

  message PhoneNumber {
    optional string number = 1;
    optional PhoneType type = 2 [default = HOME];
  }

  repeated PhoneNumber phones = 4;
}

message AddressBook {
  repeated Person people = 1;
}
```

**第二步: 根据"数据结构"自动生成代码**

```bash
# 得到 addressbook_pb2.py
protoc --python_out=./ ./addressbook.proto
```

备注: 如果使用者直接拿到这个`.py`文件, 实际上就已经可以进行第三步

**第三步: 利用生成的 .py 文件操作数据**

```python
import addressbook_pb2
from google.protobuf.json_format import MessageToDict, MessageToJson, ParseDict, Parse
import json

address_book = addressbook_pb2.AddressBook()
person = address_book.people.add()
person.id = 123
person.name = "123"
phone = person.phones.add()
phone.number = "123"

# 序列化为特定格式的字符串
s = address_book.SerializeToString()
with open("data", "wb") as fw:
    fw.write(s)
with open("data", "rb") as fr:
    s = fr.read()
address_book = addressbook_pb2.AddressBook()
# 从序列化的字符串解析protobuf message
address_book.ParseFromString(s)

# 转换为json字符串
json_str = MessageToJson(address_book)
dict_obj = json.loads(json_str)
# 转换为字典
dict_obj = MessageToDict(address_book)

# 从字典转为protobuf message
address_book = ParseDict(dict_obj, addressbook_pb2.AddressBook())

# 从json字符串转为protobuf message
address_book = Parse(json_str, addressbook_pb2.AddressBook())
```

### onnx 概念

参考[官方文档](https://onnx.ai/onnx/intro/concepts.html)，此处仅简要列出：

- node: 即张量op
  - attribute: 张量op的参数, 例如: [Unsqueeze](https://github.com/onnx/onnx/blob/main/docs/Changelog.md#unsqueeze-11) 算子(opset11版本) 中的 `axes` 就是 attribute, 构建计算图时就会确定好具体的取值
- input: 即张量op的输入
  - initializer: 一种特殊的输入, 固定的权重
- output: 即张量op的输出
- domain: onnx用domain将op进行划分(即domain是一些op的集合), 官方只定义了如下几个domain:
  - `ai.onnx`: 包含 Add, Conv, Relu 等
  - `ai.onnx.ml`: 包含 TreeEnsembleRegressor, SVMRegressor 等
  - `ai.onnx.preview.training`: onnx v1.7.0 新特性, 包含 Adam 等
- graph: 使用node, input, output搭建的图
- opset version:
  - opset version: 可以通过以下方式查看当前版本的onnx的opset版本号
  ```
  import onnx
  print(onnx.__version__, " opset=", onnx.defs.onnx_opset_version())
  # 1.13.0  opset= 18
  ```
  - op version: 每个op都有自己的版本号, 例如: Add 操作有 1, 6, 7, 13, 14这几个版本号, 这代表 Add 操作随着 opset 更新的版本
  - 一个graph会为每个domain记录一个全局的opset版本号，graph内的所有node都会按照所在的domain的opset版本号决定其版本号, 例如一个graph里设定的的ai.onnx这个domain的opset版本号为8, 则 Add 操作的版本号为 7
- proto: 上述概念实现上采用了Protocol Buffer, `onnx` 定义了如下的[数据结构](https://onnx.ai/onnx/api/classes.html)
  - 核心: `TensorProto`, `TensorShapeProto`, `TypeProto`, `ValueInfoProto`, `AttributeProto`, `OperatorProto`, `FunctionProto`, `NodeProto`, `GraphProto`, `ModelProto`, `TrainingInfoProto`
  - 其他: `MapProto`, `OperatorSetIdProto`, `OperatorSetProto`, `OptionalProto`, `SequenceProto`, `SparseTensorProto`, `StringStringEntryProto`, 

下面是一个具体的例子: 

首先参考[例子](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/notebooks/PyTorch_Bert-Squad_OnnxRuntime_GPU.ipynb)使用 `torch.onnx.export` 转换模型得到一个 `.onnx` 格式的文件

```python
# pip install transformers
# 对bert-base-uncased模型的配置做了一些修改: 词表缩小为133, embedding与隐层维数缩小为16, transformer block数量缩小为2
from transformers import BertForMaskedLM, BertTokenizer
import torch
device = "cpu"  # 使用cpu即可
model = BertForMaskedLM.from_pretrained("my-small-model").to(device)
tokenizer = BertTokenizer.from_pretrained("my-small-model")
inputs = tokenizer(["hello world"], padding="max_length", max_length=128, truncation=True, return_tensors="pt").to(device)

model.eval()
with torch.no_grad():
  symbolic_names = {0: "batch_size", 1: "max_seq_len"}
  torch.onnx.export(
    model,
    f="model.onnx",
    args=tuple(inputs.values()),
    opset_version=11,
    do_constant_folding=True,
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["last_hidden_state"],
    dynamic_axes={
      "input_ids": symbolic_names,
      "attention_mask": symbolic_names,
      "token_type_ids": symbolic_names
    }
  )
```

然后读取并解析

```python
import onnx
import google.protobuf.json_format
model_proto: onnx.ModelProto = onnx.load("model.onnx")

# onnx.load() 本质上等同于:
# from onnx import ModelProto              # ModelProto在源码中定义在
# manual_model_proto = ModelProto()
# x = open("model.onnx", "rb").read()      # byte类型
# manual_model_proto.ParseFromString(x)
# model_proto == manual_model_proto        # True

d = google.protobuf.json_format.MessageToDict(model_proto)
```

`d` 的内容如下:

```json
{
    "irVersion": 6,
    "producerName": "pytorch",
    "producerVersion": "1.9",
    "opsetImport": [{"version": "11"}],
    "graph": {
        "name": "torch-jit-export",
        "input": [
            {
                "name": "input_ids",
                "type": {
                    "tensorType": {
                        "elemType": 7,  // int64
                        "shape": {"dim": [{"dimParam": "batch_size"}, {"dimParam": "max_seq_len"}]}
                    }
                }
            }  // "token_type_ids", "attention_mask" 类似
        ],
        "output": [
            {
                "name": "last_hidden_state",
                "type": {
                    "tensorType": {
                        "elemType": 1,  // float32
                        "shape": {"dim": [{"dimParam": "batch_size"}, {"dimParam": "Addlast_hidden_state_dim_1"}, {"dimValue": "133"}]}
                    }
                }
            }
        ],
        "initializer": [
            {
                "dims": ["1", "128"],
                "dataType": 7,  // int64
                "name": "bert.embeddings.position_ids",
                "rawData": "AAAAAAABBDCCCCC"
            },
            {
                "dims": ["133", "16"],
                "dataType": 1,  // float32
                "name": "bert.embeddings.word_embeddings.weight",
                "rawData": "AAAAAAABBDCCCCC"
            }
            // ...
        ],
        "node": [  // 一共有242个算子(BertforMaskedLM的num_layers被设置为2的情况下)
            {
                "input": ["attention_mask"],
                "output": ["46"],
                "name": "Unsqueeze_0",
                "opType": "Unsqueeze",
                "attribute": [
                    {"name": "axes", "ints": ["1"], "type": "INTS"}
                ]
            },  // (B, L) -> (B, 1, L), 最终目标是匹配: (B, num_head, L, L)
            {
                "input": ["46"],
                "output": ["47"],
                "name": "Unsqueeze_1",
                "opType": "Unsqueeze",
                "attribute": [
                    {"name": "axes", "ints": ["2"], "type": "INTS"}
                ]
            },  // (B, 1, L) -> (B, 1, 1, L), 目标是匹配是匹配: (B, num_head, L, L)
            // 上面两个算子对应与transformers中的实现为 extended_attention_mask=attention_mask[:, None, None, :]
            {
                "input": ["47"],
                "output": ["48"],
                "name": "Cast_2",
                "opType": "Cast",
                "attribute": [
                    {"name": "to", "i": "1", "type": "INTS"}
                ]
            },  // 这个对应于 extended_attention_mask=extended_attention_mask.to(torch.float32)
            // 以下两个算子对应 extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min 的前半部分
            {
                "output": ["49"],
                "name": "Constant_3",
                "opType": "Constant",
                "attribute": [
                    {"name": "value", "t": {"dataType": 1, "rawData": "AACAPw=="}, "type": "TENSOR"}
                ]
            },
            {
                "input": ["49", "48"],
                "output": ["50"],
                "name": "Sub_4",
                "opType": "Sub"
            }
            // ...
        ]
    }
}
```

备注: "AACAPw==" 代表 32 位浮点数的原因可以参考这个[issue](https://github.com/onnx/onnx/issues/5244)。

```python
import base64
s = "AACAPw=="
v = base64.b64decode(s.encode())      # b'\x00\x00\x80?' => 1.0 的 IEEE754 表示: 00111111 10000000 00000000 00000000
value = np.frombuffer(v, np.float32)  # np.array([1.0])
```

### onnx Python API: (low level)

onnx定义模型的方式是使用 `*Proto` 的方式进行的：

### 源码安装解析
此处结合 make, cmake, pybind11, setup.py, protocol buffer 对 onnx 项目的安装过程以及一些使用时的调用栈进行分析

### onnxruntime

#### 安装

如果下载预编译包, `onnxruntime` 与 `onnxruntime-gpu` 不能同时安装

备注：
- onnxruntime-gpu==1.6.0 Pypi 预编译包不带 TensorrtProvider
- onnxruntime-gpu==1.10.0 Pypi 预编译包包含 TensorrtProvider
- 无论哪种情况都要注意 `onnxruntime/capi/_ld_preload.py`
  ```python
  # onnxruntime-gpu==1.10.0 onnxruntime/capi/_ld_preload.py 文件内容
  # 注意这些 cudnn 与 tensorrt 的动态链接库要包含在系统目录中
  from ctypes import CDLL, RTLD_GLOBAL
  try:
      _libcublas = CDLL("libcublas.so.11", mode=RTLD_GLOBAL)
      _libcudnn = CDLL("libcudnn.so.8", mode=RTLD_GLOBAL)
      _libcurand = CDLL("libcurand.so.10", mode=RTLD_GLOBAL)
      _libcufft = CDLL("libcufft.so.10", mode=RTLD_GLOBAL)
      _libcudart = CDLL("libcudart.so.11.0", mode=RTLD_GLOBAL)
  except OSError:
      import os
      os.environ["ORT_CUDA_UNAVAILABLE"] = "1"
  from ctypes import CDLL, RTLD_GLOBAL
  try:
      _libcudnn = CDLL("libcudnn.so.8", mode=RTLD_GLOBAL)
      _libcudart = CDLL("libcudart.so.11.0", mode=RTLD_GLOBAL)
      _libnvinfer = CDLL("libnvinfer.so.8", mode=RTLD_GLOBAL)
      _libnvinfer_plugin = CDLL("libnvinfer_plugin.so.8", mode=RTLD_GLOBAL)
  except OSError:
      import os
      os.environ["ORT_TENSORRT_UNAVAILABLE"] = "1"
  ```


#### 一个例子
本节主要对官方的[示例代码](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/notebooks/PyTorch_Bert-Squad_OnnxRuntime_GPU.ipynb)

```bash
python -m onnxruntime.transformers.optimizer \
  --input onnx.model \
  --output onnx_opt.model
  # 其余参数 ...
```

本质上干了两件事


#### 源码安装解析

这里以 TensorRT ([文档](https://fs-eire.github.io/onnxruntime/docs/build/eps.html#tensorrt), [Dockerfile](https://github.com/microsoft/onnxruntime/blob/main/dockerfiles/Dockerfile.tensorrt)) 举例。

这里结合官方[Dockerfile](https://github.com/microsoft/onnxruntime/blob/v1.6.0/dockerfiles/Dockerfile.tensorrt)简述一下 `onnxruntime-gpu==1.6.0` 的源码安装步骤, 官方的 Dockerfile 的内容简化为如下:

```dockerfile
FROM nvcr.io/nvidia/tensorrt:20.07.1-py3
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:/code/cmake-3.14.3-Linux-x86_64/bin:/opt/miniconda/bin:${PATH}
ENV LD_LIBRARY_PATH /opt/miniconda/lib:$LD_LIBRARY_PATH
RUN git clone --single-branch --branch v1.6.0 --recursive https://github.com/Microsoft/onnxruntime && \
    # apt install python3-dev
    # install miniconda
    # pip install numpy
    # install cmake
    ./build.sh --cuda_home /usr/local/cuda --cudnn_home /usr/lib/x86_64-linux-gnu/ --use_tensorrt --tensorrt_home /workspace/tensorrt --config Release --build_wheel --update --build --cmake_extra_defines ONNXRUNTIME_VERSION=1.6.0 && \
    pip install /code/onnxruntime/build/Linux/Release/dist/*.whl
```

镜像的 build 命令为

```bash
docker build -t onnxruntime-trt -f Dockerfile.tensorrt .
```

首先对基础镜像 `nvcr.io/nvidia/tensorrt:20.10-py3` (`nvcr.io/nvidia/tensorrt:20.07.1-py3` 应该类似) 做一些说明, 该基础镜像包含 CUDA、cuDNN、TensorRT, 相关信息如下:

```bash
# 环境变量
CUDA_PATH=""
CUDA_HOME=""
C_PATH=""
C_INCLUDE_PATH=""
CPLUS_INCLUDE_PATH=""
# /usr/local/nvidia/bin 目录实际不存在
PATH="/opt/tensorrt/bin:/usr/local/mpi/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin"
# LD_LIBRARY_PATH 的几个目录实际上都不存在
LD_LIBRARY_PATH="/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
# 这个目录存放有 libcublas.so 等动态链接库文件
LIBRARY_PATH="/usr/local/cuda/lib64/stubs:"
```

默认动态链接库目录
```
# cat /etc/ld.so.conf.d/*
/usr/local/cuda/compat/lib                    # 00-cuda-compat.conf(此目录不存在)
/usr/local/cuda-11.1/targets/x86_64-linux/lib # 999_cuda-11-1.conf(cuda动态链接库目录)
/usr/local/cuda/lib64                         # cuda.conf(实际上软连接到/usr/local/cuda-11.1/targets/x86_64-linux/lib)
/usr/local/lib                                # libc.conf
/usr/local/nvidia/lib                         # nvidia.conf(此目录不存在)
/usr/local/nvidia/lib64                       # nvidia.conf(此目录不存在)
/usr/local/mpi/lib                            # openmpi.conf
/usr/local/ucx/lib                            # openucx.conf
/usr/local/lib/x86_64-linux-gnu               # x86_64-linux-gnu.conf(此目录不存在)
/lib/x86_64-linux-gnu                         # x86_64-linux-gnu.conf
/usr/lib/x86_64-linux-gnu                     # x86_64-linux-gnu.conf(包含cuDNN动态链接库)
```

gcc默认头文件目录
```
# gcc -v -E -
/usr/lib/gcc/x86_64-linux-gnu/7/include
/usr/local/include
/usr/lib/gcc/x86_64-linux-gnu/7/include-fixed
/usr/include/x86_64-linux-gnu                 # 包含 cudnn_v8.h 等 cudnn 头文件以及 NvInfer.h 等 TensorRT 头文件目录
/usr/include                                  # 包含 cudnn.h 头文件, 本质上软链接到 /usr/include/x86_64-linux-gnu/cudnn_v8.h
# /usr/include/linux 目录下有一个cuda.h文件, 但没有更多的例如 curand.h 文件, 但这个目录似乎不在gcc的默认搜索路径下

nvcc默认头文件库
假设nvcc位于/usr/local/cuda/bin/nvcc
那么nvcc --verbose xx.cu 会显示出搜索的头文件信息
默认头文件搜索路径为 /usr/local/cuda/bin/../include
```

CUDA、cuDNN、TensorRT

```
CUDA
安装路径为 /usr/local/cuda, 包含 include, lib64, bin 目录

可执行文件目录 /usr/local/cuda/bin 被添加到 PATH 环境变量中 (例如: nvcc)
头文件目录 /usr/local/cuda/include 在设置了上述 PATH 变量后, 是nvcc的默认头文件目录 (例如: cublas.h)
库文件目录 /usr/local/cuda/lib64 被包含在默认动态链接库中 (例如: libcublas.so)
/usr/local/cuda/lib64/stubs 被添加到 LIBRARY_PATH 环境变量中

cuDNN
无可执行文件
头文件在默认头文件目录 /usr/include 中 (例如: cudnn.h)
库文件在默认动态链接库目录 /usr/lib/x86_64-linux-gnu 中 (例如: libcudnn.so)

TensorRT
可执行文件目录 /opt/tensorrt/bin 被添加到 PATH 环境变量中 (例如: trtexec)
头文件在默认头文件目录 /usr/include/x86_64-linux-gnu 中 (例如: NvInfer.h)
库文件在默认动态链接库目录 /usr/lib/x86_64-linux-gnu 中 (例如: libnvinfer.so)
```

源码安装的关键命令为
```bash
./build.sh \
  # 可以通过设置环境变量 CUDA_HOME 或 --cuda_home 指定, /usr/local/cuda 要包含 bin, lib64, include 目录, nvcc 所在目录需包含在环境变量 PATH 中
  --cuda_home /usr/local/cuda \
  # 可以通过设置环境变量 CUDNN_HOME 或 --cudnn_home 指定, /workspace/cudnn 包含 lib64, include 目录即可 
  --cudnn_home /workspace/cudnn \
  # 可以通过设置环境变量 TENSORRT_HOME 或 --tensorrt_home 指定, /workspace/TensorRT-7.1.3.4 包含 lib, include 目录也可
  --use_tensorrt --tensorrt_home /workspace/TensorRT-7.1.3.4 \
  # --skip_submodule_sync \  # 跳过submodule同步
  --config Release --build_wheel --update --build --cmake_extra_defines ONNXRUNTIME_VERSION=1.6.0

# cmake/CMakeLists.txt 文件中有这种写法, PATH_SUFFIXES 表示会搜索 TENSORRT_ROOT 与 TENSORRT_ROOT/include 目录
# find_path(TENSORRT_INCLUDE_DIR NvInfer.h
#   HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
#   PATH_SUFFIXES include)

# MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
# find_library(TENSORRT_LIBRARY_INFER nvinfer
#   HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
#   PATH_SUFFIXES lib lib64 lib/x64)
```