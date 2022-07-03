# Python 环境搭建

## Ubuntu 源码安装 Python

综合参考[博客](https://linuxize.com/post/how-to-install-python-3-7-on-ubuntu-18-04/)以及[csdn博客](https://blog.csdn.net/xietansheng/article/details/84791703)源码安装 Python：

```bash
sudo apt update
sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev libffi-dev wget libbz2-dev
wget https://www.python.org/ftp/python/3.7.4/Python-3.7.4.tgz
tar -xf Python-3.7.4.tgz
cd Python-3.7.4
./configure --enable-optimizations --prefix=/usr/local/python3.7
make -j 8
sudo make altinstall
ln -s /usr/local/python3.7/bin/python3.7 /usr/local/bin/python3.7
ln -s /usr/local/python3.7/bin/pip3.7 /usr/local/bin/pip3.7
```

可以在 `~/.bashrc` 或 `/etc/profile` 中添加 Python 安装的可执行文件到 PATH 环境变量。

```bash
export PATH="$PATH:/usr/local/python3.7/bin/"
```

```bash
$ . ~/bashrc  # 需重启
$ . /etc/profile  # 无需重启
```

某些用 pip 安装的包会在 `setup.py` 文件中的 `setup` 函数中指定 `scripts` 参数，这些脚本将被复制到 `/usr/local/python3.7/bin` 目录下。例如：pdfminer 的源码中

```
setup(..., scripts = ["tools/pdf2txt.py", "tools/dumppdf.py"], ...)
```

## Ubuntu apt 安装 Python

```bash
apt clean && apt update && apt upgrade -y
apt install -y software-properties-common 
# python3 available (python3 ~> python3.6)
# python3.6 -m pip, pip, pip3 not available

add-apt-repository ppa:deadsnakes/ppa -y
apt update
apt install -y python3.8  # 3.8.12 installed
# python3.8 -m pip, pip, pip3 not available

apt install -y python3-pip
python3.8 -m pip install --upgrade pip
```

在 Linux 上，时常会出现多个 python，尤其是在不熟悉的机器上，可能会发现

```
python
python3
python3.6
python3.7
python2.7
```

这些命令同时存在，此时可以使用 update-alternatives 命令对其进行管理，作为最佳实践，会将 python 2.x版本全部”绑定“到 `/usr/bin/python` 上，将 python 3.x 版本全部绑定到 `/usr/bin/python3` 上。具体过程如下

```bash
update-alternatives --install /usr/local/bin/python3 python3 /usr/bin/python3.6 1
update-alternatives --install /usr/local/bin/python3 python3 /usr/bin/python3.7 2
```

这样便形成了一个软链接 `/usr/local/bin/python3 -> /etc/alternatives/python3`，并且 python 3.7 的优先级高于 python 3.6。此时 python3 代表的就是 python3.7 了。关于 update-alternatives 的具体说明参考 [update-alternatives](https://linuxhint.com/update_alternatives_ubuntu/)。

备注：此处推荐将目标地址设定在 `/usr/local/bin/python3` 而不在 `/usr/bin/python3`，因为按 Linux 的目录结构规范，`/usr/bin` 目录应该由包管理器 apt 来管理，而 `/usr/local/bin` 是由 root 用户来手动管理的。

备注：python 2.x 的官方维护期限为2020年1月1号，因此新版本的系统上可能不会再使用 python 2.x 了，因此也可以用 update-alternatives 将 python 3.x 管理起来后，再 `/usr/local/bin/python` 直接软链接到 `/usr/local/bin/python3`。

## Python 程序的运行方式

[参考链接\(realpython.com\)](https://realpython.com/run-python-scripts/)

命令行的启动方式主要有以下两种

```bash
python -m xx.yy  # 用脚本的方式启动
python ./xx/yy.py  # 用模块的方式启动
```

两者的区别可用例子看出

例子：

```
ROOT
  - b/
    - b.py
    - __init__.py
  - c/
    - c.py
    - __init__.py
```

```python
# b.py 内容如下, 其余文件均为空
print(__name__)
import os
print(os.path.abspath(os.getcwd()))
import sys
print(sys.path)
from c import c
```

在与 `b`、`c` 目录同级的目录下启动

```bash
$ python b/b.py  # error
__main__
/home/name/ROOT
[/home/name/ROOT/b, ...]
ModuleNotFoundError: No module named 'c'
```

```bash
$ python -m b.b  # OK
__main__
/home/name/ROOT
[/home/name/ROOT, ...]
```

总结：

相同点：

- `os.getcwd()` 对于两种启动方式是一致的，以运行命令的位置一致
- 两者都将启动的 `.py` 文件的 `__name__` 赋值为 `"__main__"`

不同点：

- 采用 `python b/c/b.py` 运行时，`sys.path` 会将 `b/c` 目录添加，而 `python -m b.c.b` 会将当前目录添加到 `sys.path` 中

除此以外，使用 `python -m b.b` 运行时还有两个特殊之处：

- 如果 `b.b` 是已经用 `pip install` 安装的包名，则可以在任意目录使用该方式运行脚本，无论该脚本是否被加入 `setup.py` 文件的 `setup.py` 的 `entry_points` 参数中。
  
  例如：
  
  torch 1.9.0 版本可以用如下方式启动多卡运行脚本
  
  ```bash
  python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr 127.0.0.1 --master_port 29500 train.py
  ```
  
  ```python
  # setup.py(torch源码中entry_point并无此脚本)
  entry_points = {
          'console_scripts': [
              'convert-caffe2-to-onnx = caffe2.python.onnx.bin.conversion:caffe2_to_onnx',
              'convert-onnx-to-caffe2 = caffe2.python.onnx.bin.conversion:onnx_to_caffe2',
          ]
      }
  ```

- 以 `python -m b.b` 运行时，`b/b.py` 中的相对路径导入可以部分起作用，例如：`from .s import s`，但不能使用 `from ..c import c`，否则会报错
  
  ```
  ValueError: attempted relative import beyond top-level package
  ```
  
  若切换到 `b` 目录，以 `python b.py` 而言，即使使用 `from .s import s`，仍然会直接报错
  
  ```
  ImportError: attempted relative import with no known parent package
  ```

## 调试

pudb 调试快捷键

pdb 调试

使用 VSCode 调试 Python 代码的 launch.json 文件模板

```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: load_detr.py",
            "type": "python",
            "request": "launch",
            "program": "load_detr.py",
            "console": "integratedTerminal",
            "justMyCode": false,
            "cwd": "${workspaceFolder}",
            "args": []
        }
    ]
}
```

## Python编程规范

[参考链接](https://blog.csdn.net/u014636245/article/details/89813732)（待整理）

### 1. 命名规范

| 用途      | 命名原则 | 例子  |
|:------- |:---- |:--- |
| 类       |      |     |
| 函数/类的方法 |      |     |
| 模块名     |      |     |
| 变量名     |      |     |
|         |      |     |
|         |      |     |
|         |      |     |

### 2. 其他

```python
a = ()  # 空元组, 注意不能写为(,)
a = (1,)  # 一个元素的元组, 注意不能写为(1), 否则`a`是一个整型数字1
a = []  # 空列表, 注意不能写为[,]
# 不要使用\换行, 可以用`()`, `[]`, `{}`形成隐式换行, 注意用这两种换行方式时第二行缩进多少是任意的, 

# 列表与元组最后是否以逗号结尾要看具体情况
a = ["a",
    "b",]
a = ["a", "b"]
```

### 3. 注解的规范

[python PEP 484](https://www.python.org/dev/peps/pep-0484/)

注意 Python 解释器不会真的按照注解来检查输入输出，这些信息只是为了方便程序员理解代码、代码文档自动生成以及 IDE 的自动提示。

```python
def f(a: int = 1, b: "string" = "") -> str:
    a: int = 1
    b: "str" = "a"
    print(a, b)
a: int = 1
f.__annotations__
```

如果不使用这种方式进行注解，还可以利用 `.pyi` 文件。这种 `.pyi` 文件被称为存根文件（stub fiile），类似与 C 语言中的函数声明，详情可参考 [stackoverflow](https://stackoverflow.com/questions/59051631/what-is-the-use-of-stub-files-pyi-in-python) 问答，例如：

```python
# pkg.py文件内容
def foo(x, y):
    return x + y
# pkg.pyi文件内容，注意省略号是语法的一部分
def foo(x: int, y: int) -> int: ...
```

这种 `.pyi` 文件除了用于注释普通 `.py` 文件外，通常也用来注释 Python 包中引入的 C 代码。例如在 Pytorch 1.9.0 中，在 `torch/_C` 目录下就有许多 `.pyi` 文件，但注意这并不是 stub file，例如 `torch/_C/__init__.pyi` 文件关于 `torch.version` 的注解如下：

```python
# Defined in torch/csrc/Device.cpp
class device:
    type: str  # THPDevice_type
    index: _int  # THPDevice_index

    def __get__(self, instance, owner=None) -> device: ...

    # THPDevice_pynew
    @overload
    def __init__(self, device: Union[_device, _int, str]) -> None: ...

    @overload
    def __init__(self, type: str, index: _int) -> None: ...

    def __reduce__(self) -> Tuple[Any, ...]: ...  # THPDevice_reduce
```

关于注解的模块主要是 typing

```python
# Iterable等也在这里
from typing import List
print(isinstance([], List))  # True
# 注意List不能实例化
List([1])  # TypeError: Type List cannot be instantiated; use list() instead

# collections模块内也有Iterable
from collections.abc import Iterable
isinstance([], Iterable)  # True
```

### 4. 避免pycharm中shadows name "xxx" from outer scope的警告

以下是两个典型的情形\(注意: 这两段代码从语法及运行上说是完全正确的\)

```python
data = [4, 5, 6]
def print_data(data):  # <-- Warning: "Shadows 'data' from outer scope
    print data
print_data(data)
```

```python
# test1.py
def foo(i: int):
    print(i + 100)
# test2.py
from test1 import foo
def bar(foo: int):  # <-- Warning: "Shadows 'foo' from outer scope
    print(foo)
bar(1)
foo(10)
```

修改方式: 将形式参数重命名即可

为何要做这种规范\([参考stackoverflow回答](https://stackoverflow.com/questions/20125172/how-bad-is-shadowing-names-defined-in-outer-scopes)\): 以第一段代码为例, 假设print\_data内部语句很多, 在开发过程中突然想将形式参数`data`重命名为`d`, 但可能会由于疏忽漏改了函数内部的某个`data`, 这样代码会出现不可预料的错误, 有时难以发现\(相对于这种情形: 假设一开始将形式参数命名为`d`, 现在希望将形式参数命名为`c`, 结果由于疏忽漏改了某个`d`, 这样程序会立刻报错\). 当然, 许多IDE对重命名做的很完善, 减少了上述错误发生的可能性.

### 5. 工具

配合 Git 的使用规范如下（[官方文档](https://pre-commit.com/)，[知乎简易指南](https://zhuanlan.zhihu.com/p/65820736)）

- 安装：`pip install pre-commit`

- 在与 `.git` 目录同级的目录下新建一个 `.pre-commit-config.yaml`

- 在与 `.git` 目录同级的目录下运行
  
  ```
  pre-commit install
  ```
  
  这条命令将对 `.git/hooks` 目录新创建一个 `pre-commit` 文件，其内容是一个可执行脚本

- 执行 `git commit` 时会自动触发上述钩子

## pip

### 修改pip/conda镜像源

[参考链接](https://www.cnblogs.com/wqpkita/p/7248525.html)

**单次使用**

```text
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**永久使用**

修改相关文件的内容如下，没有的话创建一个。

```text
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
trusted-host=mirrors.aliyun.com
```

windows下的文件为`C:\Users\54120\pip\pip.ini`，linux下的文件为`~/.pip/pip.conf`

conda国内镜像方式为:

[参考链接](https://blog.csdn.net/qq_29007291/article/details/81103603?utm_medium=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase%20)

window下

```text
# 清华
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/

# 中科大
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/msys2/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/bioconda/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/menpo/
conda config --set show_channel_urls yes

# 设置搜索时显示通道地址
conda config --set show_channel_urls yes
```

linux下

vim ~/.condarc

```text
channels:
  - https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
  - https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - defaults
show_channel_urls: true
```

### pip命令

```bash
# 查看pip缓存目录
pip cache dir
# 修改pip缓存目录, 配置文件位置为"C:\\Users\\用户名\\AppData\\Roaming\\pip\\pip.ini"
pip config set global.cache-dir "D:\\Anaconda\\pipDownload\\pip\\cache"
# 用于拷贝环境
pip freeze > requirements.txt
pip install -r requirements.txt
```

### 离线安装python包

平时用的 pip install 命令的执行逻辑是如果需要下载，则先执行 pip download，再进行安装。因此大部分情况下，将 `pip install` 替换为 `pip download -d <dirname>` 即可实现只下载安装包而不安装。

**有网环境下载安装包**

```text
# 下载单个离线包
pip download -d <your_offline_packages_dir> <package_name>
# 批量下载离线包
pip download -d <your_offline_packages_dir> -r requirements.txt
```

如果无网环境的设备与有网环境的设备平台不一致（例如外网电脑是 Windows 系统，内网电脑是 Linux Ubuntu 系统），需要增加额外参数例如 `platform`、`abi`、`python-version` 等。详情可以参考 [stackoverflow 问答](https://stackoverflow.com/questions/49672621/what-are-the-valid-values-for-platform-abi-and-implementation-for-pip-do)以及它推荐的[链接](https://www.python.org/dev/peps/pep-0425/#Use)。例如：

```bash
pip download --platform win_amd64 --abi none --python-version 37 --implementation cp --only-binary=:all: -d py-package-ubuntu/ numpy
# platform指操作系统
# abi不清楚，设成none应该都是ok的
# python-version是指python版本, 37表示3.7
# implementation是指编译python的方式，例如cp表示CPython
# only-binary=:all: 是前四者设置的时候必须加上的
# 一般情况下，只要注意修改platform、python-version即可
```

其中，`platform` 参数的具体值可以在无网环境下使用如下方式得到（注意：要将所有的 `.` 与 `-` 替换为 `_`。）

```python
from distutils import util
print(util.get_platform())
```

**将文件拷贝至无网环境安装**

```text
# 安装单个离线包
pip install --no-index --find-links=<your_offline_packages_dir> <package_name>
# 批量安装离线包
pip install --no-index --find-links=<your_offline_packages_dir> -r requirements.txt
```

### 疑难杂症

在 Windows 下有时会因为权限问题，在执行

```
c
```

时，因为没有安装权限，导致原有的 `pip` 被卸载而更新的 `pip` 又不能正常安装。此时可以使用如下方式恢复

```
python -m ensurepip
```

之后保证权限后正常更新 pip 即可

```
pip install --upgrade pip
# 备用方法
# python -m pip install -U --force-reinstall pip
```

## conda 使用

### 基本操作

创建环境

```bash
conda create --name <env_name> python=<version>
# 例子：conda create --name temp python=3.8
```

删除环境

```bash
conda env remove --name <env_name>
# 例子：conda env remove --name temp
```

查看所有环境

```bash
conda env list
```

进入终端后自动激活conda base环境
```bash
conda init bash  # 自动激活
conda config --set auto_activate_base false  # 取消自动激活
```

### 不同环境设定不同的环境变量

以设定 cuda 相关的环境变量为例，以 Windows 为例，linux 类似。相关的官方文档参见[此处](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)。只需手动增加两个文件即可：

例如：`CONDA_ROOT` 为 anaconda 的安装路径，例如 `D:/anaconda3`，而 `ENV_NAME` 为环境名，例如：`tf2.7`。例如：`tf2.7` 表明此环境下需要安装 tensorflow 2.7.0 版本。其依赖关系从官网上可以查到，依赖：

- 支持 cuda 11.2 以上版本的显卡驱动

- cuda 11.2

- cudnn 8.1.0

为此，需要手动安装好相应版本的 cuda 与 cudnn，两者均可以安装多个版本。为了使得进入 `tf2.7` 环境时，自动选择 cuda 11.2 及相应版本的 cudnn，只需要配置如下两个文件即可。作用时进入 `tf2.7` 环境时，设定好 `%PATH%` 变量，退出时恢复原本的 `%PATH%`。

`${CONDA_ROOT}/envs/${ENV_NAME}/etc/conda/activate.d/activate.bat`

```shell
@echo off
set OLDPATH=%PATH%
set CUDA_VERSION=v11.2
set CUDNN_VERSION=cudnn-11.3-windows10-x64-v8.2.1.32

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%CUDA_VERSION%
set NVCUDASAMPLES_ROOT=C:\ProgramData\NVIDIA Corporation\CUDA Samples\%CUDA_VERSION%
set OTHER_CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%CUDA_VERSION%\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%CUDA_VERSION%\extras\CUPTI\lib64;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%CUDA_VERSION%\include;C:\cudnn\%CUDNN_VERSION%\cuda\bin

set PATH=%CUDA_PATH%;%NVCUDASAMPLES_ROOT%;%OTHER_CUDA_PATH%;%OLDPATH%
echo set CUDA_VERSION=%CUDA_VERSION%, CUDNN_VERSION=%CUDNN_VERSION%
```

`${CONDA_ROOT}/envs/${ENV_NAME}/etc/conda/deactivate.d/deactivate.bat`

```shell
@echo off
set PATH=%OLDPATH%
echo recover PATH
```

## Ipython在终端的使用

使用`ipython`启动, 如果要在一个cell中输入多行, 则可以使用`ctrl+o`快捷键, 注意不要连续使用两个`enter`或者在最后一行输入`enter`, 否则会使得当前cell被运行

[一个不那么好的教程](https://www.xspdf.com/resolution/50080150.html)

## jupyter使用

### kernel添加与删除

以conda管理为例, 假设需要将环境temp加入到jupyter中, 首先执行:

```bash
# 为temp环境安装ipykernel包
conda activate temp
pip install ipykernel # conda install ipykernel
```

接下来继续将temp加入至jupyter的kernel中:

```bash
jupyter kernelspec list  # 列出当前可用的kernel环境
jupyter kernelspec remove 环境名称  # 移除kernel环境
# 进入需要加入至kernel的环境后
python -m ipykernel install --user --name 环境名称 --display-name "jupyter中显示的名称"
```

使用:

```bash
# 激活base环境后
cd 目录名
jupyter-notebook # jupyter-lab
```

### 命令模式快捷键

当光标停留某个block里面的时候, 可以按下`Esc`键进入命令模式, 命令模式下的快捷键主要有:

`A`: 在上方插入一个block, `B`: 在下方插入一个block

## VSCode

```
{
    "python.formatting.provider": "yapf"
}
```

备注：

- yapf，autopep8，black 均为将代码格式化的工具

快捷键

`alt+左方向键`：跳转回退
