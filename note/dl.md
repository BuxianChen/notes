## 环境配置

说明：高版本的显卡驱动兼容各个低版本的 CUDA 与 CUDNN，但 CUDA 与 CUDNN 之间存在的对应关系。CUDA 的各个版本之间一般不存在兼容性。而CUDNN实际上是一堆头文件和库文件的集合，直接放入CUDA的头文件和库文件中即可。安装顺序为：先安装最新版本的显卡驱动，再安装CUDA，再查询相匹配版本的CUDNN，并将文件复制进CUDA中即可。

### 驱动
安装驱动的方式有几种：
- ubuntu下使用命令安装：
    ```bash
    # apt search nvidia-driver
    apt install nvidia-driver-515
    ```
    此方法的好处是不需要关闭其他可能引起冲突的系统驱动，坏处是apt维护的nvidia-driver可能不是最新的不能满足需求
- 去[官方nvidia驱动下载页面](https://www.nvidia.com/download/index.aspx)下载安装脚本进行安装
- 系统级别安装CUDA时，可以将驱动也一并勾选进行安装

### CUDA与CUDNN
安装多个版本的 CUDA 及 CUDNN：

- [参考链接](https://towardsdatascience.com/installing-multiple-cuda-cudnn-versions-in-ubuntu-fcb6aa5194e2)
- 李沐[B站课程](https://www.bilibili.com/video/BV1LT411F77M)
    主要有两种方式(除去docker的方式)：
    - 系统级别安装nvidia驱动+系统级手动安装所需要的版本的cuda
        备注：此方法得到的cuda是完整的
    - 系统级别安装nvidia驱动+conda安装cudatoolkit
        备注：此方法得到的cuda是不完整的
    
如果只希望运行Pytorch代码，可以只安装驱动，其余的CUDA、CUDNN等包含在`pip install pytorch`中[pytorch问答](https://discuss.pytorch.org/t/how-to-check-if-torch-uses-cudnn/21933)。具体来说，在官方预编译好[pytorch whl](https://download.pytorch.org/whl/torch/)包中，例如：`torch-1.11.0+cu115-cp38-cp38-linux_x86_64.whl`，将其解压后，可以发现这些文件的存在：
```
./torch/lib/libcudnn.so.8
./torch/lib/libcudnn_adv_infer.so.8
./torch/lib/libcudnn_adv_train.so.8
./torch/lib/libcudnn_cnn_infer.so.8
./torch/lib/libcudnn_cnn_train.so.8
./torch/lib/libcudnn_ops_infer.so.8
./torch/lib/libcudnn_ops_train.so.8
```
备注：pytorch中带着的cuda以及cudnn头文件及库文件并不是完整的cuda与cudnn库文件。并且即使系统级别安装的是别的版本的cuda与cudnn，使用pip的方式安装时，pytorch也会将其忽略。


### cuda driver version与runtime version

```text
nvidia-smi  # CUDA Drive Version
nvcc -V  # CUDA Runtime Version
```

CUDA 提供了两套 API，一套是 CUDA Driver API，另一套是 CUDA Runtime API，其中 Driver API 更为底层。

```
import torch
torch.version.cuda()  # Runtime version
```

一般而言，只需要关注 Runtime API 即可，例如在 Linux 系统中的 `~/.bashrc` 中添加这几行（添加到 PATH 与 LD_LIBRARY_PATH 尤为重要）指的是 Runtime API 相关的路径。

```
export CUDA_HOME=/usr/local/cuda-10.0
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib:$CUDA_HOME/lib64
```

## 日志

```
nohup python -u main.py > nohup.out 2>&1 &
```

- nohup 表示关闭终端不受影响
- `-u` 表示输出到 `nohup.out` 时不进行缓存操作
- `2>&1` 表示将标准错误与标准输出（即 `print` 函数打印的内容）重定向至 `nohup.out`
- `&` 表示任务在后台进行

对于一个前台运行的程序，可以使用`ctrl+z` 快捷键暂停，或者新开一个终端，执行如下命令让它暂停

```
kill -SIGSTOP <pid>
```

暂停任务再次启动的方法为

```
jobs  # 查看后台任务
fg %1  # 将任务1放到前台继续执行
bg %1  # 将任务1放到后台继续执行
```

或者使用

```
kill -SIGCONT 
```



## 可视化

## 有效利用GPU

## 小工具

```python
import numpy as np
# 输出两位小数, 抑制科学计数法输出
np.set_printoptions(precision=2, suppress=True)
```

捕获 NaN

```
x = torch.tensor([1, 2, np.nan])
torch.isnan(x)
```

## 训练

iteration 1

先设置一个小模型，例如：模型为一层的 LSTM；尝试过拟合一个小数据（注意去除所有例如打乱 dataset 等随机因素），优化器使用 Adam（默认参数）

