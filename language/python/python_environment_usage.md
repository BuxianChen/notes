# Python 环境搭建

## Ubuntu 源码安装 Python

综合参考[博客](https://linuxize.com/post/how-to-install-python-3-7-on-ubuntu-18-04/)以及[csdn博客](https://blog.csdn.net/xietansheng/article/details/84791703)源码安装 Python：

```bash
sudo apt update
sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev libffi-dev wget libbz2-dev
wget https://www.python.org/ftp/python/3.7.4/Python-3.7.4.tgz
tar -xf Python-3.7.4.tgz
cd Python-3.7.4
./configure --enable-optimizations --prefix=/usr/python3.7
make -j 8
sudo make altinstall
ln -s /usr/python3.7/bin/python3.7 /usr/bin/python3.7
ln -s /usr/python3.7/bin/pip3.7 /usr/local/bin/pip3.7
```

可以在 `~/.bashrc` 或 `/etc/profile` 中添加 Python 安装的可执行文件到 PATH 环境变量。

```bash
export PATH="$PATH:/usr/python3.7/bin/"
```

```bash
$ . ~/bashrc  # 需重启
$ . /ect/profile  # 无需重启
```

某些用 pip 安装的包会在 `setup.py` 文件中的 `setup` 函数中指定 `scripts` 参数，这些脚本将被复制到 `/usr/python3.7/bin` 目录下。例如：pdfminer 的源码中

```
setup(..., scripts = ["tools/pdf2txt.py", "tools/dumppdf.py"], ...)
```

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

## Ipython在终端的使用

使用`ipython`启动, 如果要在一个cell中输入多行, 则可以使用`ctrl+o`快捷键, 注意不要连续使用两个`enter`或者在最后一行输入`enter`, 否则会使得当前cell被运行

[一个不那么好的教程](https://www.xspdf.com/resolution/50080150.html)

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

## conda 使用

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

