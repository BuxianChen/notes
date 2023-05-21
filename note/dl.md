# CUDA 环境配置

说明：高版本的显卡驱动兼容各个低版本的 CUDA 与 CUDNN，但 CUDA 与 CUDNN 之间存在的对应关系。CUDA 的各个版本之间一般不存在兼容性。而CUDNN实际上是一堆头文件和库文件的集合，直接放入CUDA的头文件和库文件中即可。安装顺序为：先安装最新版本的显卡驱动，再安装CUDA，再查询相匹配版本的CUDNN，并将文件复制进CUDA中即可。


## (硬件)显卡支持

显卡与CUDA版本的支持对应情况: 参考[博客](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

## (系统软件)驱动
安装驱动的方式有几种：
- ubuntu下使用命令安装：
    ```bash
    # apt search nvidia-driver
    apt install nvidia-driver-515
    ```
    此方法的好处是不需要关闭其他可能引起冲突的系统驱动，坏处是apt维护的nvidia-driver可能不是最新的不能满足需求
- 去[官方nvidia驱动下载页面](https://www.nvidia.com/download/index.aspx)下载安装脚本进行安装
- 系统级别安装CUDA时，可以将驱动也一并勾选进行安装

## (依赖软件) gcc, g++

安装 CUDA 时需要执行安装脚本，可能会涉及到 gcc/g++ 不存在或不兼容的情况，安装多个版本的 gcc/g++(ubuntu): 参考[博客](https://linuxconfig.org/how-to-switch-between-multiple-gcc-and-g-compiler-versions-on-ubuntu-20-04-lts-focal-fossa)


## TensorRT

相关下载及文档链接如下：

- CUDA Driver: https://developer.nvidia.com/download/indec.aspx
- CUDA版本与驱动的支持关系: https://developer.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
- CUDA:
  - 下载及文档：https://developer.nvidia.com/cuda-toolkit-archive
- cuDNN:
  - 下载：https://developer.nvidia.com/rdp/cudnn-archive
- TensorRT: 
  - 下载：https://developer.nvidia.com/nvidia-tensorrt-download
  - 文档：https://docs.nvidia.com/deeplearning/tensorrt/archives/index.html

TensorRT 依赖于 CUDA 与 cuDNN, 且特定版本的 TensorRT 依赖于特定版本的 CUDA 与 cuDNN，下面是几个版本的例子：

TensorRT 8.2 GA + cuDNN 8.2.4 + CUDA 11.4
- TensorRT 8.2 GA Update 4 for Linux x86_64 and CUDA 11.0, 11.1, 11.2, 11.3, 11.4 and 11.5 TAR Package: TensorRT-8.2.5.1.Linux.x86_64-gnu.cuda-11.4.cudnn8.2.tar.gz
- cuDNN v8.2.4 (September 2nd, 2021), for CUDA 11.4, for Linux (x86): cudnn-11.4-linux-x64-v8.2.4.15.tgz

TensorRT 8.2 GA + cuDNN 8.2.4 + CUDA 10.2
- TensorRT 8.2 GA Update 4 for Linux x86_64 and CUDA 10.2 TAR Package:TensorRT-8.2.5.1.Linux.x86_64-gnu.cuda-10.2.cudnn8.2.tar.gz
- cuDNN v8.2.4 (September 2nd, 2021), for CUDA 10.2, for Linux (x86): cudnn-10.2-linux-x64-v8.2.4.15.tgz

TensorRT 7.2.3 + cuDNN 8.1.1 + CUDA 11.1
- TensorRT 7.2.3 for Ubuntu 18.04 and CUDA 11.1, 11.2 TAR package: TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.1.tar.gz
- cuDNN v8.1.1 (Feburary 26th, 2021), for CUDA 11.0, 11.1, 11.2, for Linux Ubuntu(x86_64): cudnn-11.2-linux-x64-v8.1.1.33.tgz


TensorRT 7.2.3 + cuDNN 8.1.1 + CUDA 10.2
- TensorRT 7.2.3 for Ubuntu 18.04 and CUDA 10.2 TAR package: TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.1.tar.gz
- cuDNN v8.1.1 (Feburary 26th, 2021), for CUDA 10.2, for Linux Ubuntu(x86_64): cudnn-10.2-linux-x64-v8.1.1.33.tgz


CUDA Toolkit 10.2.89 (Nov 2019):
- cuda_10.2.89_440.33.01_linux.run
- patch 1 (Aug 26, 2020): cuda_10.2.1_linux.run
- patch 2 (Nov 17, 2020): cuda_10.2.2_linux.run
CUDA Toolkit 11.4.4 (Feburary 2022): cuda_11.4.4_470.82.01_linux.run
CUDA Toolkit 11.1.1 (October 2020): cuda_11.1.1_455.32.00_linux.run

版本选择的方法【后续仍需补充】：

从下载后的 TensorRT 的安装包文件名可以看出其依赖的 CUDA 与 cuDNN 版本
从下载后的 cuDNN 的安装包文件名可以看出其依赖的 CUDA 版本
从下载后的 CUDA 的安装包文件名可以看出与其相适配的显卡驱动版本号（如果机器本身的显卡驱动足够高，安装时可以选择不安装显卡驱动）

## 多版本并存安装指南

参考资料

- [博客](https://towardsdatascience.com/installing-multiple-cuda-cudnn-versions-in-ubuntu-fcb6aa5194e2)
- 李沐[B站课程](https://www.bilibili.com/video/BV1LT411F77M)
    主要有两种方式(除去docker的方式)：
    - 系统级别安装nvidia驱动+系统级手动安装所需要的版本的cuda
        备注：此方法得到的cuda是完整的
    - 系统级别安装nvidia驱动+conda安装cudatoolkit
        备注：此方法得到的cuda是不完整的

本节内容仅作宏观指导, 具体细节以官方文档为准

第一步：安装版本足够高的显卡驱动，驱动决定的支持最高的 CUDA 版本，且驱动在宿主机上只能是唯一的版本

CUDA、cuDNN、TensorRT 都可以多版本并存安装，其中 CUDA 需要执行安装脚本、而 cuDNN 和 TensorRT 只需要将下载的安装文件解压并设置好环境变量即可。

下面以这个版本为例介绍安装方法：

TensorRT 7.2.3 + cuDNN 8.1.1 + CUDA 10.2
- TensorRT 7.2.3 for Ubuntu 18.04 and CUDA 10.2 TAR package: TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.1.tar.gz
- cuDNN v8.1.1 (Feburary 26th, 2021), for CUDA 10.2, for Linux Ubuntu(x86_64): cudnn-10.2-linux-x64-v8.1.1.33.tgz
- CUDA Toolkit 10.2.89 (Nov 2019):
  - cuda_10.2.89_440.33.01_linux.run
  - patch 1 (Aug 26, 2020): cuda_10.2.1_linux.run
  - patch 2 (Nov 17, 2020): cuda_10.2.2_linux.run


即假设目录 `/home/username/software` 下下载好了如下文件
```
cuda_10.2.89_440.33.01_linux.run
cuda_10.2.1_linux.run
cuda_10.2.2_linux.run
cudnn-10.2-linux-x64-v8.1.1.33.tgz
TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.1.tar.gz
```

**CUDA 安装**
参考[博客](https://blog.csdn.net/qq757056521/article/details/109267381)

```bash
# 在希望不修改/usr/local/cuda的软连接时
./cuda_10.2.89_440.33.01_linux.run --librarypath=/usr/local/cuda-10.2
./cuda_10.2.1_linux.run  # 安装 patch1
./cuda_10.2.2_linux.run  # 安装 patch2
# 安装完后可以将上面 3 个安装脚本文件全部删除
```

以上命令会将 CUDA 10.2 安装在 `/usr/local/cuda-10.2` 目录下

需要修改的环境变量有：

```bash
export PATH=/usr/local/cuda-10.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH
export CPATH=/usr/local/cuda-10.2/include:$CPATH
```

**cuDNN 安装**

参考 [stackoverflow](https://stackoverflow.com/questions/40736999/installing-cudnn-for-general-usage-without-root-access)
```bash
tar -zxvf cudnn-10.2-linux-x64-v8.1.1.33.tgz
# 官网以及许多教程会推荐将解压后的头文件与库文件直接拷贝至cuda头文件与库文件中, 这里使用分离的办法
mv cuda cudnn-10.2-linux-x64-v8.1.1.33
rm -rf cudnn-10.2-linux-x64-v8.1.1.33.tgz  # 注意不能删除解压后的文件
```

需要修改的环境变量有：
```bash
export LD_LIBRARY_PATH=/home/username/software/cudnn-10.2-linux-x64-v8.1.1.33/lib:$LD_LIBRARY_PATH
export CPATH=/home/username/software/cudnn-10.2-linux-x64-v8.1.1.33/include:$CPATH
```

**TensorRT 安装**

```bash
tar -zxvf TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.1.tar.gz
rm -rf TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.1.tar.gz  # 注意不能删除解压后的文件
# 可选: 安装python相关的包
cd TensorRT--7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-10.2/python
python3 -m pip install tensorrt-*-cp3x-none-linux_x86_64.whl
```

需要修改的环境变量有：
```bash
export PATH=/home/username/software/TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.1/bin:$PATH
export LD_LIBRARY_PATH=/home/username/software/TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.1/lib:$LD_LIBRARY_PATH
export CPATH=/home/username/software/TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.1/include:$CPATH
```

## (FAQ) CUDA 的多种安装方式

CUDA 的安装方式在网上各种资料里大略有如下几种，但一般来说只推荐上一节的那种方式，这里对这几种方式做一个说明：

- Nvidia 官网安装：
  - 离线安装脚本：例如上面介绍的 `cuda_10.2.89_440.33.01_linux.run`，这种安装包含了整个 CUDA 的工具包，最重要的是 nvcc 编译器、头文件、库文件，最全面
  - deb 包：这种安装也是全面的安装，缺点是它会把头文件、库文件、nvcc 分散在系统文件的各处，不方便删除
- 使用conda安装cuda-toolkit: 阉割版的cuda，且必须激活环境，总的来说不推荐

## (FAQ) cuda driver version与runtime version

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
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## (FAQ) Pytorch 相关

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
备注：pytorch中带着的cuda以及cudnn头文件及库文件并不是完整的cuda与cudnn库文件。并且即使系统级别安装的是别的版本的cuda与cudnn，使用pip的方式安装时，pytorch在运行时也会将其忽略【设置LD_LIBRARY_PATH会不会影响到？】。

由于pytorch自带(使用pip安装)了一个阉割版的CUDA(可能pytorch为了用户体验,只要求安装驱动其他都自带), 而某些其他的包需要独立安装一个完整的CUDA, 这时候会引发一些冲突,具体解释可以参考[参考资料](https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/trainer#trainer-integrations)

新版本的 pytorch 安装时，有一些新的变化，例如安装包变小了，CUDA和CUDNN的依赖会去下载别的nvidia的包，参见[issue](https://github.com/pytorch/pytorch/issues/96595)


pytorch 预编译包有两个地方可以下载

- [PyPi](https://pypi.org/project/torch/): 对于一个torch版本而言, 对于一个系统类型(例如:windows/linux), 一个python版本(例如:python3.6/python3.7/python3.8), 只能提供一个whl安装包, 因此只能放一个CUDA版本的预编译包
- [pytorch whl](https://download.pytorch.org/whl/torch/)：对于一个torch版本而言, Pytorch 预编译了不同 CUDA 版本的预编译包

注意: 不是所有的 CUDA 版本都有预编译包, 例如 CUDA 11.4 版本就没有预编译包

以下是在Linux环境下不同python、pytorch、cuda版本下对推理过程的测试，比较结果的差异【待测试】

- pytorch1.10.2+cuda10.2+python3.7: 一致
- pytorch1.10.2+cuda10.2+python3.8
- pytorch1.10.2+cuda10.2+python3.9
- pytorch1.12.1+cuda10.2+python3.7: 一致
- pytorch1.12.1+cuda10.2+python3.8
- pytorch1.12.1+cuda10.2+python3.9
- pytorch1.12.1+cuda11.3+python3.7: 一致
- pytorch1.12.1+cuda11.3+python3.8
- pytorch1.12.1+cuda11.3+python3.9

测试方法如下：使用ubuntu镜像预编译好3种python作为3个基础镜像, 之后使用这三个镜像分别安装pytorch whl文件, 最后在huggingface上模型结果是否有差异.

## (FAQ) 怎么知道版本号

来源: ColossalAI `colossalai check -i`
```python
# 系统的CUDA版本
from torch.utils.cpp_extension import CUDA_HOME
output = subprocess.check_output([CUDA_HOME + "/bin/nvcc", "-V"], universal_newlines=True)

# torch的版本
import torch
print(torch.__version__)

# torch内部C++算子所使用的CUDA版本
print(torch.version.cuda)
```

## (FAQ) 相关的环境变量解释

- `PATH`: 可执行文件搜索目录
- `CPATH`: 头文件搜索目录
- `CUDA_HOME`: 这个某些情况下似乎会用到, 不确定是否必须
- `LD_LIBRARY_PATH`, `RPATH`, `RUNPATH`, `DT_RPATH`, `DT_RUNPATH`: 动态链接库搜索目录

[资料1](https://medium.com/obscure-system/rpath-vs-runpath-883029b17c45)
[资料2](https://man7.org/training/download/shlib_dynlinker_slides.pdf)
[pytorch issue](https://github.com/pytorch/pytorch/issues/96595)

## (FAQ) Nvidia 显卡架构, 型号, CUDA 时间线

参考资料

- [架构](https://en.wikipedia.org/wiki/Category:Nvidia_microarchitectures)
- [CUDA](https://developer.nvidia.com/cuda-toolkit-archive)
- [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive)
- [TensorRT](https://developer.nvidia.com/nvidia-tensorrt-download)

显卡:
- 2006: Tesla
- 2010: Fermi
- 2012: Kepler
- 2014: Maxwell
- 2016: Pascal
- 2017: Volta, Tesla V100
- 2018: Turing, GeForce16 系列, RTX20 系列
- 2020: Ampere, A100, RTX3060
- 2022: Hopper, H100

CUDA:
- 2016.9: CUDA 8.0
- 2017.9: CUDA 9.0
- 2017.10: CUDA 9.1
- 2018.5: CUDA 9.2
- 2018.9: CUDA 10.0
- 2019.2: CUDA 10.1
- 2019.10: CUDA 10.2
- 2020.1: CUDA 11.0.1
- 2020.9: CUDA 11.1.0
- 2020.12: CUDA 11.2.0
- 2021.4: CUDA 11.3.0
- 2021.6: CUDA 11.4.0
- 2021.10: CUDA 11.5.0
- 2022.1: CUDA 11.6.0
- 2022.5: CUDA 11.7.0
- 2022.10: CUDA 11.8.0
- 2022.12: CUDA 12.0.0
- 2023.2: CUDA 12.1.0

cuDNN:
- 2017.4: cuDNN 6.0 (CUDA 7.5, 8.0)
- 2017.10: cuDNN 7.0.1 (CUDA 8.0, 9.0)
- 2018.3: cuDNN 7.1.2 (CUDA 9.0, 9.1, 9.2)
- 2018.8: cuDNN 7.2.1 (CUDA 9.2)
- 2018.9: cuDNN 7.3.0 (CUDA 9.0, 9.2, 10.0)
- 2018.10: cuDNN 7.4.1 (CUDA 9.0, 9.2, 10.0)
- 2019.2: cuDNN 7.5.0 (CUDA 9.2, 10.0, 10.1)
- 2019.5: cuDNN 7.6.0 (CUDA 9.2, 10.0, 10.1, 10.2)
- 2020.7: cuDNN 8.0.2 (CUDA 10.1, 10.2, 11.0)
- 2021.1: cuDNN 8.1.0 (CUDA 10.2, 11.0, 11.1, 11.2)
- 2021.4: cuDNN 8.2.0 (CUDA 10.2, 11.x, 11.4)
- 2021.10: cuDNN 8.3.0 (CUDA 10.2, 11.5)
- 2022.4: cuDNN 8.4.0 (CUDA 10.2, 11.x)
- 2022.8: cuDNN 8.5.0 (CUDA 10.2, 11.x)
- 2022.10: cuDNN 8.6.0 (CUDA 10.2, 11.x)
- 2022.11: cuDNN 8.7.0 (CUDA 10.2, 11.x)
- 2023.2: cuDNN 8.8.0 (CUDA 11.x, 12.0)
- 2023.4: cuDNN 8.9.0 (CUDA 11.x, 12.x)

TensorRT(不确定时间线在哪找)


# WSL2

前置条件：最好是 Windows 11

安装步骤：

- 按微软 [官方文档](https://docs.microsoft.com/en-us/windows/wsl/install) 安装 WSL2

- 在微软商店中安装 Ubuntu 20.04 以及 Windows Terminal (Windows 10) 已自带

后置事项：

- Windows 上安装 VSCode 并安装 Remote-WSL 插件

- 更换 apt 源：[清华源官网](https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/)

- 文件目录：
  - Windows 系统查看 WSL2 文件：在资源浏览器目录中输入：`\\wsl$`，或者类似这种目录 `C:\Users\{Username}\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu20.04LTS_79rhkp1fndgsc\LocalState` (不推荐)
  - WSL2 命令行（Windows Terminal）中查看 Windows 文件目录：`/mnt/c` 表示 C 盘目录
  - 在 WSL2 命令行中切换至相应目录执行 `code .` 即可打开 VSCode，并且打开后 VSCode 的集成终端本身也会是 WSL2 命令行

- GPU（不确定）：在 Windows 11 本机将 Nvidia 显卡驱动升级至 510 以上版本，那么不做特殊设置，WSL2 中也能访问显卡，在 WSL2 命令行中可以查验：`lspci | grep -i nvidia`

- Docker：可以参考 Docker 官方文档将 Docker Desktop 在 Windows 本机安装，此时 WSL2 与 Windows 本机均能使用 Docker。也可以在 WSL2 Terminal 中使用命令安装 Docker，这样 Docker 只能在 WSL2 命令行中访问

- 网络：一般情况下，无需做特殊设置，WSL2 的网络与 Windows 本机的网络一般是互通的。如果使用 VPN 或代理时，可能需要进行特殊设置（不确定）。

磁盘清理：

[参考资料1](https://stephenreescarter.net/how-to-shrink-a-wsl2-virtual-disk/) [参考资料2](https://stackoverflow.com/questions/64068185/docker-image-taking-up-space-after-deletion) 使用 Windows PowerShell 输入

```
diskpart
```

在弹出的命令行中输入

```powershell
# 清理WSL2
select vdisk file="C:\Users\xyz\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu20.04LTS_79rhkp1fndgsc\LocalState\ext4.vhdx"
compact vdisk
# 清理Docker Desktop
select vdisk file="C:\Users\xyz\AppData\Local\Docker\wsl\data\ext4.vhdx"
compact vdisk
select vdisk file="C:\Users\xyz\AppData\Local\Docker\wsl\distro\ext4.vhdx"
compact vdisk
```

## 远程访问WSL

在WSL2的terminal中
```bash
sudo apt remove openssh-server
sudo apt install openssh-server  # 选择1
sudo vim /etc/ssg/sshd_config
# 配置以下三项
# Port 2222
# PermitRootLogin yes
# PasswordAuthentication yes
sudo service ssh --full-restart
ifconfig  # 查看wsl2的ipv4, 以下用<wsl2_ip>代替
```

在WSL2的Windows宿主机打开powershell以管理员权限设置端口转发
```powershell
netsh interface portproxy set v4tov4 listenport=3333 listenaddress=0.0.0.0 connectport=2222 connectaddress=<wsl2_ip>
ipconfig  # 查看Windows本机ipv4, 以下用<windows_ip>代替
```

在WSL2的Windows宿主机设置防火墙规则:

控制面板->防火墙->入站规则->新建规则

- 端口
- TCP, 特定本地端口, 3333
- 允许连接
- 全选上
- 名称任意

至此, 从局域网中另一台机器打开终端即可远程连接WSL2
```
ssh <wsl2_user_name>@<windows_ip> -p 3333
```

# 源

## pip 源

参考：https://mirrors.tuna.tsinghua.edu.cn/help/pypi/

```
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## anaconda 源

参考：https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/

windows 机器先执行
```
conda config --set show_channel_urls yes
```

在~/.condarc中写入
```
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

## apt 源

参考：https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/

```
sudo sed -i "s@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
sudo sed -i "s@http://.*security.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
```

## 时区

```
export TIME_ZONE=Asia/Shanghai
apt install -y apt-utils tzdata
ln -snf /usr/share/zoneinfo/$TIME_ZONE /etc/localtime && echo $TIME_ZONE > /etc/timezone
dpkg-reconfigure -f noninteractive tzdata
```

Dockerfile中的写法
```dockerfile
ARG DEBIAN_FRONTEND="noninteractive"
ENV TIME_ZONE Asia/Shanghai
RUN apt install -y apt-utils tzdata && \
    ln -snf /usr/share/zoneinfo/$TIME_ZONE /etc/localtime && \
    echo $TIME_ZONE > /etc/timezone && \
    dpkg-reconfigure -f noninteractive tzdata
```

## virtualenvwrapper

安装
```
pip install pbr
pip install virtualenv virtualenvwrapper
```

配置
```
# 仅作示例
# ~/.bashrc
export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/opt/conda/bin/python
export VIRTUALENVWRAPPER_VIRTUALENV=/opt/conda/bin/virtualenv
source /opt/conda/bin/virtualenvwrapper.sh
```

使用
```
mkvirtualenv env-name  # 创建
rmvirtualenv env-name  # 移除
workon env-name        # 激活环境
deactivate             # 退出环境
```

备注：
- 实际上，`workon`, `mkvirtualenv` 等命令是 virtualenvwrapper.sh 这个脚本里的函数，因此使用 `whereis workon` 会找不到位置。
- 假设环境名为 `test`，则虚拟环境的 `bin` 目录为 `$WORKON_HOME/test/bin`，此目录底下包含如下可执行命令：
  ```
  python  # 一般是软链接，链接到 mkvirtualenv 时的 python 路径
  pip  # 是 pip 包的一个 entrypoint，实际上是一个文本文件（有可执行权限），注意观察第一行以#!开头，记录了执行时的python路径
  activate  # 是一个shell脚本, 注意其中有一行：VIRTUAL_ENV="~/.virtualenvs/test", 此处硬编码了虚拟环境家目录, workon大约本质上去执行了activate脚本
  ```

## jupyter添加核

```
pip install ipykernel
workon env-name
python -m ipykernel install --user --name env-name --display-name env-name
```

## ssh-key

`~/.ssh` 目录结构

```
authorized_keys  # 将其他机器的公钥写入此文件中, 则其他机器可以ssh免密登录
id_rsa  # 本机私钥
id_rsa.pub  # 本机公钥
known_hosts
```

- 为本机生成 `id_rsa` 与 `id_rsa.pub` 备用
  ```bash
  ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa
  ```

- 将本机的 `id_rsa.pub` 的内容追加到服务器特定用户的 `~/.ssh/authorized_keys` 文件内，可以实现本机到服务器的远程免密登录

  - 本地 Shell 连接服务器无需输入密码。`ssh username@ip_addr`，例如：`ssh foo@172.16.83.43`
  - VScode 远程连接无需输入密码

- 将本机的 `id_rsa.pub` 的内容在 gitlab 或 github 上添加到 SSH Keys 中，则可以免密使用 ssh 进行仓库克隆、推送等操作，例如：

  ```
  git clone git@github.com:BuxianChen/notes.git
  ```

  但对 http 的方式无效，如果使用下面的方式进行 clone，在执行 push 的时候，会自动跳出一个弹出框，要求输入 github 的帐号及密码：

  ```
  git clone https://github.com/BuxianChen/notes.git
  ```

备注：要实现远程免密登陆，服务器端 `.ssh` 文件夹的权限应该为 `700`，而 `authorized_keys` 文件的权限应为 `600`

# 日志

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

# 可视化

## tensorboard

参考资料：[Pytorch tutorial](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)、[Pytorch API](https://pytorch.org/docs/stable/tensorboard.html)

训练时将数据写入
```python
writer = SummaryWriter()
for epoch in range(4):
    for batch_idx in range(10)
        global_idx = 10 * epoch + batch_idx
        writer.add_scalar("Loss/train", loss: float, global_idx)  # 画出曲线图
        writer.add_image('a/ori_image', ori_image, global_idx)  # ori_image: (3, H, W) tensor
        writer.add_image('a/trans_image', trans_image, global_idx)  # trans_image: (3, H, W) tensor
        # 还可以写入Figure
        # writer.add_figure("matplotlib_figure", figure, global_idx)
        # 还可以对数据投影可视化
        # features: (N, C) tensor, class_labels: List[str], 长度为 N, label_img: (N, 3, H, W)
        # writer.add_embedding(features, metadata=class_labels, label_img=images)
        
        # add_video, add_audio, add_text
        # add_graph: 可视化模型(计算图)

        # add_pr_curve, add_hist: 添加PR曲线等(其实基本也可以用add_figure代替)
        # add_hparams: 这个不确定, 用于对比不同参数下实验结果?
    writer.flush()  # 将数据保存, 但不关闭
writer.close()  # 将数据保存
```

训练结束后怎么从tensorborad保存的数据中读取？

参考[stackoverflow问答](https://stackoverflow.com/questions/37304461/tensorflow-importing-data-from-a-tensorboard-tfevent-file)

```python
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
event_acc = EventAccumulator("event_file")
event_acc.Reload()
print(event_acc.Tags())

for e in event_acc.Scalars('Loss/train'):
    print(e.step, e.value)
```

## wandb（待研究）

据说这个工具相比 Tensorboard 更有优势，[参考视频](https://www.bilibili.com/video/BV17A41167WX/?spm_id_from=333.337.search-card.all.click)

# 有效利用GPU

# 小工具

## numpy 浮点数输出

```python
import numpy as np
# 输出两位小数, 抑制科学计数法输出
np.set_printoptions(precision=2, suppress=True)
```

## torch 捕获 NaN

```
x = torch.tensor([1, 2, np.nan])
torch.isnan(x)
```

## 监控 GPU 的使用

```
watch -n 1 nvidia-smi
```

或

```bash
pip install gpustat
gpustat -i 1
```

