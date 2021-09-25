# python-draft

## PART 1 环境配置

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
ln -s /usr/python3.8/bin/python3.8 /usr/bin/python3.7
ln -s /usr/python3.8/bin/pip3.8 /usr/local/bin/pip3.7
```

## Python程序的运行方式\(待补充\)

[参考链接\(realpython.com\)](https://realpython.com/run-python-scripts/)

命令行的启动方式主要有以下两种（**注意这种情况下当前目录下有xx文件夹**）

```text
python -m xx.yy
python ./xx/yy.py
```

用IDE里的按钮或快捷键来启动时，最终会回到上述两种之一，一般为第二种，但需要搞清楚**当前目录是什么**。 _待确认：_在Pycharm中，实际上是第二种方式，并且当前目录为`xx/`（即先执行了`cd xx`，再执行了`python yy.py`）

假定目录结构如下

```text
test/
  src/
      module0.py
    module1.py
    module2.py
```

各文件内容如下：

```python
# module0.py
def foo():
    print("do something")
# module1.py
from src.module0 import foo
def wrapper():
    foo()
if __name__ == "__main__":
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

```text
# 为temp环境安装ipykernel包
conda activate temp
pip install ipykernel # conda install ipykernel
```

接下来继续将temp加入至jupyter的kernel中:

```text
jupyter kernelspec list  # 列出当前可用的kernel环境
jupyter kernelspec remove 环境名称  # 移除kernel环境
# 进入需要加入至kernel的环境后
python -m ipykernel install --user --name 环境名称 --display-name "jupyter中显示的名称"
```

使用:

```text
# 激活base环境后
cd 目录名
jupyter-notebook # jupyter-lab
```

### 命令模式快捷键

当光标停留某个block里面的时候, 可以按下`Esc`键进入命令模式, 命令模式下的快捷键主要有:

`A`: 在上方插入一个block, `B`: 在下方插入一个block

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



## PART 2 Advanced Python

## Python编程规范

[参考链接](https://blog.csdn.net/u014636245/article/details/89813732)（待整理）

### 1. 命名规范

| 用途 | 命名原则 | 例子 |
| :--- | :--- | :--- |
| 类 |  |  |
| 函数/类的方法 |  |  |
| 模块名 |  |  |
| 变量名 |  |  |
|  |  |  |
|  |  |  |
|  |  |  |

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

## Python高阶

### 1. 装饰器

内置装饰器

```python
class A:
    b = 0
    def __init__(self):
        self.a = 1
    @classmethod
    def foo(cls, a):
        print(a)
    @classmethod
    def bar(cls, a):
        cls.b += a
        print(cls.b)
A.bar(3)
A.bar(2)
```

自定义装饰器

```python
from functools import wraps
def node_func(name):
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if name == "A":  # in self.nodes_df.columns:
                return 1  # dict(self.nodes_df[name])
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorate
# 等价于：foo1 = node_func("A")(foo1)
@node_func("A")
def foo1(a):
    return "a"

@node_func("B")
def bar1(a):
    return "a"
```

**`property` 装饰器**

例子来源于 [Python 官方文档](https://docs.python.org/3/library/functions.html#property)。

```python
class C:
    def __init__(self):
        self._x = None

    @property
    def x(self):
        """I'm the 'x' property."""
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @x.deleter
    def x(self):
        del self._x
```

根据前面所述，装饰器只是一个语法糖。property 函数的特征标（signature）如下：

```
property(fget=None, fset=None, fdel=None, doc=None) -> object
```

前一段代码等价于这种直接使用 `property` 函数的做法：

```python
class C:
    def __init__(self):
        self._x = None

    def getx(self):
        return self._x

    def setx(self, value):
        self._x = value

    def delx(self):
        del self._x

    x = property(getx, setx, delx, "I'm the 'x' property.")
```

备注：property 本质上是一个 Descriptor，参见后面。

### 2. 魔术方法与内置函数

#### 2.0 Python 官方文档

- 官方文档主目录：https://docs.python.org/3/
- 对 Python 语言的一般性描述：https://docs.python.org/3/reference/index.html
  - 数据模型：https://docs.python.org/3/reference/datamodel.html
- Python 标准库：https://docs.python.org/3/library/index.html
  - build-in functions（官方建议优先阅读此章节）：https://docs.python.org/3/library/functions.html
  - build-in types：https://docs.python.org/3/library/stdtypes.html
- Python HOWTOs（深入介绍一些主题，可以认为是官方博客）：https://docs.python.org/3/howto/index.html
  - Descriptor HowTo Guide：https://docs.python.org/3/howto/descriptor.html

#### 2.1 object 类

```python
>>> dir(object())
['__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__']
```

```python
class Basic:
    pass
basic = Basic()
set(dir(basic)) - set(dir(object))
# {'__dict__', '__module__', '__weakref__'}
```

**`__module__`**

**`__weakref__`**

#### 2.2 `__str__`、`__repr__` 特殊方法，str、repr 内置函数

**从设计理念上说：两者都是将对象输出，一般而言，`__str__` 遵循可读性原则，`__repr__` 遵循准确性原则。**

分别对应于内置方法 `str` 与 `repr`，二者在默认情况（不重写方法的情况下）下都会输出类似于 `<Classname object at 0x000001EA748D6DC8>` 的信息.

```python
>>> class Test:
...     def __init__(self):
...         self.a = 1
...     def __repr__(self): # 一般遵循准确性, 例如出现类似<class xxx>
...         return "__repr__"
...     def __str__(self): # 一般遵循可读性
...         return "__str__"
...
>>> test = Test()
>>> test
__repr__
>>> print(test) # print使用__str__
__str__
```

```python
>>> class Test1:
...     def __str__(self):
...         return "__str__"
...
>>> test1 = Test1()
>>> print(test1)  # print使用__str__
__str__
>>> test1
<__main__.Test1 object at 0x000001EA748D6DC8>
```

备注: 在 jupyter notebook 中, 对 `pandas` 的 `DataFrame` 使用 `print` 方法，打印出的结果不美观，但不用 `print` 却很美观，原因未知。

#### 2.3 内置函数 vars 与 `__dict__` 属性

**从设计理念上说，`vars` 函数的作用是返回对象的属性名（不会包含方法及特殊属性）。`__dict__` 属性里保存着对象的属性名（不会包含方法以及特殊属性）。这里的特殊属性指的是 `__xxx__`。**

一般情况下，Python 中的对象都有默认的 `__dict__` 属性。而 `vars(obj)` 的作用就是获取对象 `obj` 的 `__dict__` 属性。关于 `vars` 函数的解释可以参考[官方文档](https://docs.python.org/3/library/functions.html#vars)，如下：

> Return the `__dict__` attribute for a **module, class, instance, or any other object with a `__dict__` attribute**.
>
> Objects such as modules and instances have an updateable `__dict__` attribute; however, other objects may have write restrictions on their `__dict__` attributes (for example, classes use a [`types.MappingProxyType`](https://docs.python.org/3/library/types.html#types.MappingProxyType) to prevent direct dictionary updates).
>
> Without an argument, `vars()` acts like [`locals()`](https://docs.python.org/3/library/functions.html#locals). Note, the locals dictionary is only useful for reads since updates to the locals dictionary are ignored.
>
> A `TypeError` exception is raised if an object is specified but it doesn’t have a `__dict__` attribute (for example, if its class defines the [`__slots__`](https://docs.python.org/3/reference/datamodel.html#object.__slots__) attribute).

```python
# vars(x)
x.__dict__  # 必须定义为一个字典
```

备注：object 类没有 `__dict__` 属性，但继承自 object 子类的对象会有一个默认的 `__dict__` 属性（有一个例外是当该类定义了类属性 `__slots__` 时，该类的对象就不会有 `__dict__` 属性）。

**`__dict__` 属性与 Python 的查找顺序（lookup chain）息息相关，详情见 Descriptor**。

#### 2.4 `__slots__`属性

**从设计理念上说，`__slots__` 属性的作用是规定一个类只能有那些属性，防止类的实例随意地动态添加属性。**

可以定义类属性 `__slots__`（一个属性名列表），确保该类的实例不会添加 `__slots__` 以外的属性。一个副作用是定义了 `__slots__` 属性的类，其实例将不会拥有 `__dict__` 属性。具体用法如下：

```python
class A:
    __slots__ = ["a", "b"]
a = A()
a.a = 2
a.c = 3  # 报错
```

注意：假设类 `B` 继承自定义了 `__slots__` 的类 `A`，那么子类 `B` 的实例不会受到父类 `__slots__` 的限制。

#### 2.5 内置函数 dir 与 `__dir__` 方法

**从设计理念上说：不同于 vars 与 `__dict__`，dir 方法倾向于给出全部信息：包括特殊方法名**

`dir` 函数返回的是一个标识符名列表，逻辑是：首先寻找 `__dir__` 函数的定义（object 类中有着默认的实现），若存在 `__dir__` 函数，则返回 `list(x.__dir__())`。备注：`__dir__` 函数必须定义为一个可迭代对象。

若该类没有自定义 `__dir__` 函数，则使用 object 类的实现逻辑，大略如下：

> If the object does not provide [`__dir__()`](https://docs.python.org/3/reference/datamodel.html#object.__dir__), the function tries its best to gather information from the object’s [`__dict__`](https://docs.python.org/3/library/stdtypes.html#object.__dict__) attribute, if defined, and from its type object. The resulting list is not necessarily complete, and may be inaccurate when the object has a custom [`__getattr__()`](https://docs.python.org/3/reference/datamodel.html#object.__getattr__).
>
> The default [`dir()`](https://docs.python.org/3/library/functions.html?highlight=dir#dir) mechanism behaves differently with different types of objects, as it attempts to produce the most relevant, rather than complete, information:
>
> - If the object is a module object, the list contains the names of the module’s attributes.
> - If the object is a type or class object, the list contains the names of its attributes, and recursively of the attributes of its bases.
> - Otherwise, the list contains the object’s attributes’ names, the names of its class’s attributes, and recursively of the attributes of its class’s base classes.
>
> ——https://docs.python.org/3/library/functions.html

备注：官方文档对默认的 `dir` 函数的实现逻辑有些含糊不清，只能简单理解为默认实现会去寻找 `__dict__` 属性，故暂不予以深究。这里留一个测试例子待后续研究：

例子

```python
class Test:
    __slots__ = ["a", "b", "c"]
    def __init__(self):
        self.a = 3
        self.b = 1
        # self._c = 2
        # self.__d = 3
        # self.__dict__ = {"a": 1}

    def __dir__(self):
        # return "abc"
        # return {"a": "dir_a"}
        print("Test: __dir__")
        return super().__dir__()
    
    def __getattribute__(self, name: str):
        print(f"Test: __getattribute__, args: {name}")
        return super().__getattribute__(name)
    
    def __getattr__(self, name):
        print(f"Test: __gatattr__, args: {name}")
        return "default"
        # return super().__getattr__(name) # object没有__getattr__方法
test = Test()
print(dir(test))
```

输出结果为：（`__getattribute__` 与 `__getattr__` 见下一部分，大体上是寻找了 `__dict__` 属性与 `__class__` 属性）

```
Test: __dir__
Test: __getattribute__, args: __dict__
Test: __gatattr__, args: __dict__
Test: __getattribute__, args: __class__
['__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattr__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__slots__', '__str__', '__subclasshook__', 'a', 'b', 'c']
```

#### 2.6 `__getattr__`、`__getattribute__` 特殊方法，`getattr` 内置函数

**从设计理念上说，这三者的作用是使用属性名获取属性值，也适用于方法**

**作用：`__getattribute__` 会拦截所有对属性的获取。**

首先内置函数 `getattr(object, name[, default])` 的功能等同于 `object.name`，例如：`getattr(a, "name")` 等价于 `a.name`。实现细节上，内置函数 `getattr` 会首先调用 `__getattribute__`，如果找不到该属性，则去调用 `__getattr__` 函数。

备注：object 类只有 `__getattribute__` 的定义，而没有 `__getattr__`。

备注：对于以双下划线开头的变量，编译时会对其名称进行修改：

```python
class A:
	class A:
    def __init__(self):
        self.__a = 1
a = A()
dir(a)  # 会显示 "_A__a"
vars(a)  # 会显示 "_A__a"
a._A__a  # ok
getattr(a, "_A__a")  # ok
```

备注：如果要自定义 `__getattribute__` 函数，最好在其内部调用 `object.__getattribute__(self, name)`。

#### 2.7 `delattr` 内置方法、`__delattr__` 特殊方法、del 语句

**作用：`__delattr__` 会拦截所有对属性的删除。**

#### 2.8 `setattr` 内置方法、`__setattr__` 特殊方法

**作用：`__setattr__` 会拦截所有对属性的赋值。**

#### 2.9 Descriptor、`__get__`、`__set__`、`__del__`

参考： 

- [RealPython (Python-descriptors)](https://realpython.com/python-descriptors/)
- [Python 官方文档 (Howto-descriptor)](https://docs.python.org/3/howto/descriptor.html)
- [Python 官方文档 (library-build-in-functions)](https://docs.python.org/3/library/functions.html)
- [Python 官方文档 (reference-data-model)](https://docs.python.org/3/reference/datamodel.html)

**注：大多数情况下，无须使用 Descriptor**

##### 概念

按照如下要求实现了 `__get__`、`__set__`、`__delete__` 其中之一的类即满足 Descriptor 协议，称这样的类为 Descriptor（描述符） 。若没有实现 `__set__` 及 `__delete__` 方法，称为 **data descriptor**，否则称为 **non-data descriptor**。

```python
__get__(self, obj, type=None) -> object
__set__(self, obj, value) -> None
__delete__(self, obj) -> None
__set_name__(self, owner, name)
```

##### Descriptor 的作用

在 Python 的底层，`staticmethod()`、`property()`、`classmethod()`、`__slots__` 都是借助 Descriptor 实现的。



`def foo(self, *args)` 可以使用 `obj.foo(*args)` 进行调用也是使用 Descriptor 实现的。

> The starting point for descriptor invocation is a binding, `a.x`. How the arguments are assembled depends on `a`:
>
> - Direct Call
>
>   The simplest and least common call is when user code directly invokes a descriptor method: `x.__get__(a)`.
>
> - Instance Binding
>
>   If binding to an object instance, `a.x` is transformed into the call: `type(a).__dict__['x'].__get__(a, type(a))`.
>
> - Class Binding
>
>   If binding to a class, `A.x` is transformed into the call: `A.__dict__['x'].__get__(None, A)`.
>
> - Super Binding
>
>   If `a` is an instance of [`super`](https://docs.python.org/3/library/functions.html#super), then the binding `super(B, obj).m()` searches `obj.__class__.__mro__` for the base class `A` immediately preceding `B` and then invokes the descriptor with the call: `A.__dict__['m'].__get__(obj, obj.__class__)`.
>
> —— https://docs.python.org/3/reference/datamodel.html#invoking-descriptors

##### 查找顺序

完整的顺序如下，对于 `obj.x`，获得其值的查找顺序为：

- 首先寻找命名为 `x` 的 **data descriptor**。即如果在 `obj` 的类 `Obj` 定义里有如下形式：

  ```
  class Obj:
  	x = DescriptorTemplate()
  ```

  其中 `DescriptorTemplate` 中定义了 `__set__` 或 `__del__` 方法。

- 若上一条失败，在对象 `obj` 的 `__dict__` 属性中查找 `"x"`。

- 若上一条失败，寻找命名为 `x` 的 **non-data descriptor**。即如果在 `obj` 的类 `Obj` 定义里有如下形式：

  ```
  class Obj:
  	x = DescriptorTemplate()
  ```

  其中 `DescriptorTemplate` 中定义了 `__get__` 但没有定义 `__set__` 及 `__del__` 方法。

- 若上一条失败，则在 `obj` 类型的 `__dict__` 属性中查找，即 `type(obj).__dict__`。

- 若上一条失败，则在其父类中查找，即 `type(obj).__base__.__dict__`。

- 若上一条失败，则按照父类搜索顺序 `type(obj).__mro__`，对类祖先的 `__dict__` 属性依次查找。

- 若上一条失败，则得到 `AttributeError` 异常。

例子：

如果类没有定义 `__slot__` 属性及 `__getattr__` 方法，且 `__getattribute__`、`__delattr__`、`__setattr__` 这些方法都直接继承自 object 类，那么 `__dict__` 的构建将会是如下默认的方式：

```python
class Vehicle():
    can_fly = False
    number_of_weels = 0

class Car(Vehicle):
    number_of_weels = 4

    def __init__(self, color):
        self.color = color

def foo(self):
    print("foo")

my_car = Car("red")
print(my_car.__dict__)
print(type(my_car).__dict__)
my_car.bar = foo  # 注意这种情况下my_car.bar是一个unbound fuction, 关于这一点参见Descriptor
print(my_car.__dict__)
print(type(my_car).__dict__)
my_car.bar(my_car)
```

```python
{'color': 'red'}
{'__module__': '__main__', 'number_of_weels': 4, '__init__': <function Car.__init__ at 0x000001A3C7857040>, '__doc__': None}
{'color': 'red', 'bar': <function foo at 0x000001A3C76ED160>}
{'__module__': '__main__', 'number_of_weels': 4, '__init__': <function Car.__init__ at 0x000001A3C7857040>, '__doc__': None}
foo
```

查找顺序

```python
my_car = Car("red")
print(my_car.__dict__['color'])  # 等价于 mycar.color
print(type(my_car).__dict__['number_of_weels'])  # 等价于 mycar.number_of_wheels
print(type(my_car).__base__.__dict__['can_fly'])  # 等价于 mycar.can_fly
```

##### 使用 Descriptor

需实现下列函数，实现 `__get__`、`__set__`、`__delete__` 其中之一即可，`__set_name__` 为 Python 3.6 引入的新特性，可选。参照例子解释：

```python
__get__(self, obj, type=None) -> object
# self指的是Descriptor对象实例number, obj是self所依附的对象my_foo_object, type是Foo
__set__(self, obj, value) -> None
# self指的是Descriptor对象实例number, obj是self所依附的对象my_foo_object, value是3
__delete__(self, obj) -> None
# self指的是Descriptor对象实例number, obj是self所依附的对象my_foo_object
__set_name__(self, owner, name)
# self指的是Descriptor对象实例number, owner是Foo, name是"number"
```

例子

```python
class OneDigitNumericValue():
    def __set_name__(self, owner, name):
        # owner is Foo, name is number
        self.name = name

    def __get__(self, obj, type=None) -> object:
        return obj.__dict__.get(self.name) or 0

    def __set__(self, obj, value) -> None:
        obj.__dict__[self.name] = value

class Foo():
    number = OneDigitNumericValue()

my_foo_object = Foo()
my_second_foo_object = Foo()

my_foo_object.number = 3
print(my_foo_object.number)
print(my_second_foo_object.number)

my_third_foo_object = Foo()
print(my_third_foo_object.number)
```

##### 实用例子

**避免重复使用 `property`**

```python
class Values:
    def __init__(self):
        self._value1 = 0
        self._value2 = 0
        self._value3 = 0

    @property
    def value1(self):
        return self._value1

    @value1.setter
    def value1(self, value):
        self._value1 = value if value % 2 == 0 else 0

    @property
    def value2(self):
        return self._value2

    @value2.setter
    def value2(self, value):
        self._value2 = value if value % 2 == 0 else 0

    @property
    def value3(self):
        return self._value3

    @value3.setter
    def value3(self, value):
        self._value3 = value if value % 2 == 0 else 0

my_values = Values()
my_values.value1 = 1
my_values.value2 = 4
print(my_values.value1)
print(my_values.value2)
```

可以使用如下方法实现

```python
class EvenNumber:
    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, type=None) -> object:
        return obj.__dict__.get(self.name) or 0

    def __set__(self, obj, value) -> None:
        obj.__dict__[self.name] = (value if value % 2 == 0 else 0)

class Values:
    value1 = EvenNumber()
    value2 = EvenNumber()
    value3 = EvenNumber()
    
my_values = Values()
my_values.value1 = 1
my_values.value2 = 4
print(my_values.value1)
print(my_values.value2)
```

### 3. 继承

#### MRO (Method Resolution Order) 与 C3 算法

Python 在产生多继承关系时，由于子类可能有多个或多层父类，因此方法的搜索顺序（MRO, Method Resolution Order）很重要，同时，搜索顺序也涉及到类的属性。对于属性或者变量的访问，按照 MRO 的顺序依次搜索，直到找到匹配的属性或变量为止。对于每个类，可以使用如下代码来获取 MRO ：

```python
C.mro()  # C 是一个类
# 或者：
C.__mro__
```

本部分参考 C3 算法[官方文档](https://www.python.org/download/releases/2.3/mro/)：

> unless you make strong use of multiple inheritance and you have non-trivial hierarchies, you don't need to understand the C3 algorithm, and you can easily skip this paper.

**一点历史与 MRO 应满足的性质**

在 Python 的历史上，曾出现了若干种 MRO 算法，自 Python 2.3 以后，使用 C3 算法，它满足两个性质（之前的算法违背了这两个性质，所以可能会引发隐蔽的 BUG）

- local precedence ordering：MRO 的结果里应该保证父类列表的相对顺序不变。例如：

  ```python
  class A(B, C, D): pass
  ```

  MRO(A) 序列必须为 `[A, ..., B, ..., C, ..., D, ...]` 这种形式。

- monotonicity（单调性）：如果 C 的 MRO 序列中 A 排在 B 的前面，那么对于任意继承自 C 的类 D，D 的 MRO 序列中 A 也排在 B 的前面

**C3 算法**

引入记号：

- 用 $$B_1B_2...B_n$$ 代表 $$[B_1,B_2,...,B_n]$$。用 $$C+B_1...B_n$$ 代表 $$CB_1,...B_n$$。即类 $$C$$ 的 MRO 序列为 $$L(C)$$
- 对于序列 $$B_1...B_n$$，$$B_1$$ 称为头，$$B_2...B_n$$ 称为尾

C3 算法描述为：

```
L[C(B1,...,Bn)] = C + merge(L[B1],...,L[Bn], B1B2...Bn)
```

其中 merge 的规则为：

递归调用 merge 操作：

记第一个序列中的头为 $$H$$，若 $H$ 不在其余任意序列的尾中，则将 $$H$$ 添加到 MRO 序列中，并对 merge 中的所有序列中删除 $$H$$，之后对剩余序列继续 merge 操作；否则对第二个序列的头进行上述操作，直至最后一个序列。若直到最后一个序列都无法进行删除操作，那么判定为继承关系不合法。

例子：

```python
O=object
class F(O): pass
class E(O): pass
class D(O): pass
class C(D, F): pass
class B(E, D): pass
class A(B, C): pass
```

```
L[O] = O
L[F(O)] = F + merge(L[O], O) = F + merge(O, O) = FO
L[E(O)] = EO
L[D(O)] = DO
L[C(D, F)] = C + merge(L(D), L(F), DF) = C + merge(DO, FO, DF)
           = CD + merge(O, FO, F)  # D 只在所有序列的头部出现
           = CDF + merge(O, O) # O 在第二个序列的尾部出现，因此接下来对 F 进行判断
           = CDFO
L[B(E, D)] = B + merge(EO, DO, ED) = BEDO
L[A(B, C)] = A + merge(BEDO, CDFO, BC)
           = AB + merge(EDO, CDFO, C)
           = ABE + merge(DO, CDFO, C)
           = ABEC + merge(DO, DFO)
           = ABECDFO
```

#### `super` 函数

参考资料：[RealPython](https://realpython.com/python-super/)、《Python Cookbook (3ed)》chapter 8.7。



由于方法覆盖的特性，以方法为例，如果类的 MRO 顺序中有同名方法，那么处于 MRO 靠后类的同名方法将会被隐藏。因此如果需要调用父类被隐藏的方法，需要对 MRO 顺序进行调整。这就是 `super` 方法的作用。



`super` 函数有两种调用形式

- 两个参数的形式：super(cls, obj)。其中第一个参数为子类，obj 为子类对象（也可以是子类的子类对象，但基本不可能会这样去用）。

- 无参数形式：super()。推荐使用

```python
class A:
    def afoo(self):
        print("A::afoo")
class B(A):
    def afoo(self):x
        super().afoo()  # 等价于 super(B, self).afoo()
        print("B::afoo")
class C(B):
    def afoo(self):
        super(B, self).afoo()
        print("C::afoo")
C().afoo()  # 依次调用 A.afoo, C.afoo
B().afoo()  # 依次调用 A.afoo, B.afoo
```

super 实际上是一个类，但注意 `super()` 返回的不是父类对象，而是一个代理对象。

```python
class Base: def __init__(self): print("Base"); super().__init__()
class A(Base): def __init__(self): print("A"); super().__init__()
class B(Base): def __init__(self): print("B"); super().__init__()
class C(A, B): def __init__(self): print("C"); super().__init__()
C()
# 输出：
# C
# A
# B
# Base
```

上例为典型的菱形继承方式，使用 `super` 可以按照 MRO 顺序依次调用 `__init__` 函数一次。

备注：`super` 函数还有单参数的调用形式，参见 [stckoverflow](https://stackoverflow.com/questions/30190185/how-to-use-super-with-one-argument)（理解需要有许多前置知识）。

### 4. 元类

参考资料：[RealPython](https://realpython.com/python-metaclasses/)，[Python 官方文档](https://docs.python.org/3/reference/datamodel.html#metaclasses)，

类是用来构造实例的，因此类也可以被叫做实例工厂；同样地，也有构造类的东西，被称为**元类**。实际上每个类都需要用元类来构造，默认的元类为 `type`。

```python
class A: pass
# 等同于
class A(object, metaclass=type): pass
```

#### `type` 函数

Python 中, type 函数是一个特殊的函数，调用形式有两种：

- `type(obj)`：返回 obj 的类型
- `type()`



#### `__new__` 函数与 `__init__` 函数（待补充）

#### `abc` 模块

`abc` 模块最常见是搭配使用 `ABCMeta` 与 `abstractmethod`。其作用是让子类必须重写父类用 `abstractmethod` 装饰的方法，否则在创建子类对象时就会报错。[参考](https://riptutorial.com/python/example/23083/why-how-to-use-abcmeta-and--abstractmethod)

用法如下：

```python
from abc import ABCMeta, abstractmethod
class Base(metaclass=ABCMeta):
    @abstractmethod
    def foo(self):
        print("foo")
    @abstractmethod
    def bar(self):
        pass
class A(Base):
    def foo(self):
        print("A foo")
    def bar(self):
        print("A bar")
a = A()
super(A, a).foo()
a.foo()
a.bar()
```

注意：不设定 `metaclass=ABCMeta` 时，`abstractmethod` 不起作用，即不会强制子类继承。

使用 `ABCMeta` 与 `abstractmethod` 优于这种写法：

```python
class Base(metaclass=ABCMeta):
    def foo(self):
        print("a foo")
    def bar(self):
        raise NotImplementedError()
class A(Base):
    def foo(self):
        print("A foo")
a = A()
super(A, a).foo()
a.foo()
a.bar()  # 此时才会抛出异常
```

### 5. with语法\(含少量contextlib包的笔记\)

主要是为了理解pytorch以及tensorflow中各种with语句

主要[参考链接](https://www.geeksforgeeks.org/with-statement-in-python/)

#### 5.1 读写文件的例子

首先厘清读写文件的一些细节

```python
# test01.py
file = open("record.txt", "w+")
file.write("Hello")  # 由于file没有调用close方法, 所以"Hello"未被写入
file = open("record.txt", "w+")
file.write("World")
file.close()  # 这一行是否有都是一样的, 大概是解释器自动调用了close
# 这个脚本最终只会写入"World"
```

以下三段代码中

* 代码1如果在write时报错, 那么文件无法被close, 有可能引发BUG
* 代码2保证文件会被close, 另外可以通过增加except语句, 使得可以处理各类异常
* 代码3则相对优雅, 并且与代码2功能一致, 即使write出错, close依旧会被调用

```python
# 1) without using with statement
file = open('file_path', 'w')
file.write('hello world !')
file.close()

# 2) without using with statement
file = open('file_path', 'w')
try:
    file.write('hello world')
finally:
    file.close()

# 3) using with statement
with open('file_path', 'w') as file:
    file.write('hello world !')
```

代码3是怎么做到的呢? 其实际上基本等效于

```python
foo = open("file_path", "w")
file = foo.__enter__()
try:
    file.write("hello world !")
finally:
    # 注意: 此处需要传递3个参数, 但一般不会是None
    foo.__exit__(None, None, None)
```

注意到一般情况下, 此处的foo与file是不一样的对象, 参见下节中关于`__enter__`方法的返回值. 但在文件读写的情形下, foo与file是相同的对象. 另外, `__exit__`函数有三个参数, 在自定义这个函数时也应该遵循三个参数的设计\(具体可以参考[这个问答](https://www.reddit.com/r/learnprogramming/comments/duvc2r/problem_with_classes_and_with_statement_in_python/)\).

#### 5.2 with语法与怎么让自定义类支持with语法

> This interface of \_\_enter\_\_\(\) and \_\_exit\_\_\(\) methods which provides the support of with statement in user defined objects is called `Context Manager`.

总的来说, 需要让类支持with语法, 只需要定义魔术方法`__enter__`与`__exit__`即可, 一个完整的例子如下

```python
class A():
    def __init__(self):
        print("create A")
    def do_before_enter(self):
        print("do before exit")
        self.a = 1
    def __enter__(self):
        self.do_before_enter()
        print("__enter__")
        return self.a  # 如果使用with A() as x形式, 此处的返回值由x接收
    def __exit__(self, exc_type, exc_value, traceback):
        self.do_before_exit()
        print("__exit__")
    def do_before_exit(self):
        print("do before exit")
        del self.a

x = A()
print(hasattr(x, "a"))  # False
with x as a:
    print(hasattr(x, "a"))  # True
    print(x is a)  # False
    print(f"run with block, a: {a}")
    # 取消下一行的注释, __exit__方法依然会被调用
    # xxx(f"run with block, a: {a}")
print(hasattr(x, "a"))  # False

# 忽略异常处理, 基本等同于如下代码段
# x = A()
# a = x.__enter__()
# print(f"run with block, a: {a}")
# x.__exit__(None, None, None)
```

#### \*5.3 使用contextlib包中的函数来使得类支持with语法

按照上一节的做法, 可以使用如下写法让`MassageWriter`支持with语法

```python
class MessageWriter(object):
    def __init__(self, file_name):
        self.file_name = file_name

    def __enter__(self):
        self.file = open(self.file_name, 'w')
        return self.file

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()

with MessageWriter('my_file.txt') as xfile:
    xfile.write('hello world')
```

也可以使用`contextlib`中的一些方法不进行显式定义`__enter__`与`__exit__`使得自定义类能支持with语法, 例子如下

```python
from contextlib import contextmanager

class MessageWriter(object):
    def __init__(self, filename):
        self.file_name = filename

    # 此处需要定义为生成器而不能是函数
    @contextmanager
    def open_file(self):
        try:
            file = open(self.file_name, 'w')
            yield file
        finally:
            file.close()

message_writer = MessageWriter('record.txt')
with message_writer.open_file() as my_file:
    my_file.write('Hello world')
```

执行顺序为: 首先`open_file`函数被调用, 并且将返回值`file`传递给`my_file`, 之后执行with语句内部的`write`方法, 之后再回到open\_file方法的`yeild file`后继续执行. 可以简单理解为:

* open_file函数从第一个语句直到第一个yield语句为\`\_enter_\`
* open_file函数从第一个yield语句到最后为\`\_exit_\`

#### 5.4 "复合"with语句

```python
with open(in_path) as fr, open(out_path, "w") as fw:
    pass
```

```python
from contextlib import ExitStack
import csv
def rel2logic(in_path, logic_dir):
    """将关系表转为逻辑客关系表形式
    Example:
        >>> rel2logic("./python_logical/tests/all_relations.tsv", "./python_logical/tests/gen")
    """
    with ExitStack() as stack:
        fr = csv.DictReader(stack.enter_context(open(in_path, encoding="utf-8")), delimiter="\t")
        fws = {}
        for row in fr:
            start_type, end_type = row["start_type"], row["end_type"]
            start_id, end_id, relation = row["start_id"], row["end_id"], row["relation"]
            key = start_type + "-" + end_type + ".tsv"
            if key not in fws:
                out_path = os.path.join(logic_dir, key)
                fw = stack.enter_context(open(out_path, "w", encoding="utf-8"))
                fws[key] = csv.writer(fw, delimiter="\t", lineterminator="\n")
                fws[key].writerow([start_type, end_type, "relation"])
            fws[key].writerow([start_id, end_id, relation])
```

### 6. for else语法

```python
# 获取[1, n]中的所有素数
for n in range(2, 10):
    for x in range(2, n):
        if n % x == 0:
            print( n, 'equals', x, '*', n/x)
            break
    else:
        # loop fell through without finding a factor
        print(n, 'is a prime number')
# 来源于Cython文档里的例子
```

### 7. python基本数据类型

int: 无限精度整数

float: 通常利用`C`里的`double`来实现

### 8. 函数的参数

参考[知乎](https://www.zhihu.com/question/57726430/answer/818740295)

**函数调用**

```
funcname(【位置实参】,【关键字实参】)
```

使用了 `a=x` 这种方式传参的即为关键字实参。

两个具有一般形式的例子

```python
# 1, 2 为位置实参，
foo(1, 2, a=3, b=4)  # 一般调用形式
foo(1, *[0], 2, *[3, 4], a=1, **{"c": 1}, **{"d": 1})  # 特殊调用形式
```

**函数定义**

```
def funcname(【限定位置形参】,【普通形参】,【特殊形参args】,【限定关键字形参】,【特殊形参kwargs】): pass
```

<font color=red>备注：限定位置形参在 Python 3.8 才被正式引入，即 `/` 这种写法。在此之前仅有后面的四种形参</font>

一个具有一般形式的例子：

```python
def foo(a, b, /, c, d=3, *args, e=5, f, **kwargs): pass
def foo(a, b=1, /, c=2, d=3, *, e=5, f, **kwargs): pass
```

- `a` 与 `b` 为限定位置形参
- `c` 与 `d` 为普通形参
- `e` 与 `f` 为限定关键字形参

**形实结合的具体过程**

首先用位置实参依次匹配限定位置形参和普通形参，其中位置实参的个数必须大于等于限定位置形参的个数，剩余的位置实参依顺序匹配普通形参。

- 若位置实参匹配完全部限定位置形参和普通形参后还有剩余，则将剩余参数放入 `args` 中
- 若位置实参匹配不能匹配完全部普通形参，则未匹配上的普通形参留待后续处理

接下来用关键字实参匹配普通形参和限定关键字形参，匹配方式按参数名匹配即可。

**设定默认值的规则**

为形参设定默认值的规则与前面的规则是独立的。

- 限定关键字形参，带默认值与不带默认值的形参顺序随意
- 限定位置形参和普通形参，带默认值的形参必须位于不带默认值的形参之后

### 9. 导包规则

参考：

- [RealPython: python-import](https://realpython.com/python-import/)
- [RealPython: modules-packages-introduction](https://realpython.com/python-modules-packages)
- [Real Python: Namespaces and Scope in Python](https://realpython.com/python-namespaces-scope/)

首先，需要厘清几个概念：

- namespace

  ```
  import a
  print(a.xxx)
  ```

  这里的 `a` 是一个 `namespace`

- module

  单个 `.py` 文件是一个 `module`

- package

  目录，且目录下有 `__init__.py` 文件

- namespace package

  目录，且目录下没有 `__init__.py` 文件

#### 9.1 namespace

- built-in namespace (运行脚本里的变量)
- global namespace
- enclosing namespace (带有内层函数的函数)
- local namespace (函数最里面的一层)

```python
>>> globals()  # 返回global namespace
'__name__': '__main__', '__doc__': None, '__package__': None, '__loader__': <class '_frozen_importlib.BuiltinImporter'>, '__spec__': None, '__annotations__': {}, '__builtins__': <module 
'builtins' (built-in)>
>>> locals()  # 返回local/enclosing namespace, 当位于运行脚本时, 与globals结果一样
```

```python
global x, y  # 声明当前作用域下引用的是全局变量x, y
nonlocal x, y  # 声明当前作用域下引用的是上一层作用域的x, y
```

```python
from xx import yy
# 将yy引入当前作用域, sys.modules中会有xx模块

import xx.yy  # xx必须是一个包, yy可以是一个包或模块
# sys.modules会显示xx是一个namespace, xx.yy是一个模块
# globals() 只包含xx, 不包含yy及xx.yy

import .xx  # 不允许
```

global namespace 需要额外进行说明，与 import 相关。

#### 9.2 import 语法详解

**绝对导入与相对导入**

```python
# 绝对导入
from aa import bb
from aa.bb import C
import aa.bb  # aa.bb 必须为一个module/namespace package/package

# 相对导入：必须以点开头，且只有from ... import ...这一种写法
from . import aa
from ..aa.bb import cc
# import .aa  # 无此语法
```

**`from ... import ...` 语法详解**

下面分别对上述导入语句作解析：

```
from aa import bb
```

导入成功只能为三种情况

- `aa` 是一个不带 `__init__.py` 的文件夹（namespace package）。

  - `bb` 是一个 `bb.py` 文件。则可以直接使用 `bb`，但不能使用 `aa` 以及 `aa.bb`。注意，此时

  ```python
  sys.modules["aa"]  # 显示为namespace
  sys.modules["aa.bb"]  # 显示为module
  sys.modules["bb"]  # 报错
  ```

  - `bb` 是一个带或者不带 `__init__.py` 的文件夹，情况类似，唯一的区别是此时 `bb` 会显示为一个 module 或者是 namespace。

- `aa` 是一个带有 `__init__.py` 的文件夹（package），则上述导入成功的条件为 `bb` 在 `aa/__init__.py` 中是一个标识符，或者 `bb` 是 `aa` 的子目录，或者 `bb.py` 在文件夹 `aa` 下。无论是哪种情况，`aa/__init__.py` 均会被执行，且 `aa` 与 `aa.bb` 不可直接使用。下面是一个例子：

  目录结构为

  ```
  aa
    - __init__.py
    - bb.py
  ```

  文件内容为

  ```python
  
  # aa/__init__.py
  c = 1
  print(c)
  # bb.py
  # 无内容
  ```

  使用

  ```python
  >>> from aa import bb  # 注意此时已经将c打印了
  1
  >>> bb
  <module 'aa.bb' from 'aa/bb.py'>
  >>> # aa.cc, aa, aa.bb # 三者均不可使用
  >>> import aa  # 注意aa/__init__.py不会再次被执行
  >>> aa.bb
  <module 'aa.bb' from 'aa/bb.py'>
  >>> aa.c
  1
  >>> aa
  <module 'aa.bb' from 'aa/__init__.py'>
  ```

- `aa` 是一个 `aa.py` 文件，则上述导入成功的条件为 `aa.py` 中可以使用 `bb` 这一标识符。

```
from aa.bb import C
```

结论：对于这种形式的导入

```
from xx import yy
from xx.yy import zz
```

`xx.py` 或 `xx/__init__.py` 只要有就会被执行。并且 `xx` 与 `yy` 是 namespace package 还是 package 不影响导入，最终只有 import 后面的东西可以直接使用。

**`import ...` 语法详解**

```python
import aa.bb.cc
```

导入成功只能为一种情况 `aa/bb/cc.py` 或着 `aa/bb/cc` 存在，作用是依次执行 `aa/__init__.py`，`aa/bb/__init__.py`，`aa/bb/cc.__init__.py` （若它们都是package）。无论 `aa` 与 `bb` 是 package/namespace package，以下标识符均可以直接使用：

```
aa
aa.bb
aa.bb.cc
aa.foo  # foo 在 aa/__init__.py 中
aa.bb.bar  # bar 在 bb/__init__.py 中
```

以下不可使用

```
aa.zz  # aa/zz.py文件, 且aa/__init__.py中没有from . import zz
```

备注：无论是 `from ... import ...` 还是 `import ...`，相关包的 `__init__.py` 及 `xx.py` 模块均会被执行一次。后续若再次 import，无论文件是否发生变动，均不会再次运行 `__init__.py` 或 `xx.py` 文件。只是标识符是否可用发生变化。

**彻底理解import**

**step1：官方文档搜索记录**

平时惯用的 import 语法是 `importlib.__import__` 函数的语法糖：

> The [`__import__()`](https://docs.python.org/3/library/importlib.html?highlight=import#importlib.__import__) function
>
> ​		The [`import`](https://docs.python.org/3/reference/simple_stmts.html#import) statement is syntactic sugar for this function
> ——https://docs.python.org/3/library/importlib.html

其函数定义为（[链接](https://docs.python.org/3/library/importlib.html?highlight=import#importlib.__import__)）：

```python
importlib.__import__(name, globals=None, locals=None, fromlist=(), level=0)
```

官方对此函数的解释为：

> An implementation of the built-in [`__import__()`](https://docs.python.org/3/library/functions.html#__import__) function.
>
> Note: Programmatic importing of modules should use [`import_module()`](https://docs.python.org/3/library/importlib.html?highlight=import#importlib.import_module) instead of this function.

即：`importlib.__import__` 是内置函数的一种实现。备注：此处官方的超链接疑似有误，似乎应该是：**平时惯用的 import 语法是内置函数 `__import__` 函数的语法糖**

而内置函数 `__import__` 的定义为（[链接](https://docs.python.org/3/library/functions.html#__import__)）：

```python
__import__(name, globals=None, locals=None, fromlist=(), level=0)
```

官方对此函数有如下注解：

> This function is invoked by the [`import`](https://docs.python.org/3/reference/simple_stmts.html#import) statement.  It can be replaced (by importing the [`builtins`](https://docs.python.org/3/library/builtins.html#module-builtins) module and assigning to `builtins.__import__`) in order to change semantics of the `import` statement, but doing so is **strongly** discouraged as it is usually simpler to use import hooks (see [**PEP 302**](https://www.python.org/dev/peps/pep-0302)) to attain the same goals and does not cause issues with code which assumes the default import implementation is in use.  Direct use of [`__import__()`](https://docs.python.org/3/library/functions.html#__import__) is also discouraged in favor of [`importlib.import_module()`](https://docs.python.org/3/library/importlib.html#importlib.import_module).

可以看到，`importlib.__import__` 与内置函数 `__import__` 的定义完全相同。

总结：平时所用的 import 语句仅仅是 `importlib.__import__` 函数（也许是内置函数 `__import__`）的语法糖。而 `importlib.__import__` 是内置函数 `__import__` 的一种实现，建议不要直接使用 `importlib.__import__` 与内置的 `__import__` 函数。

整理一下官方说明链接：

- `import` 语法：[链接1](https://docs.python.org/3/reference/simple_stmts.html#import)
- `importlib.__import__` 函数：[链接2](https://docs.python.org/3/library/importlib.html#importlib.__import__)
- `__importlib__` 内置函数：[链接3](https://docs.python.org/3/library/functions.html#__import__)

由于 `importlib.__import__` 函数几乎没有任何说明，因此主要看链接 1 与 3。

**step 2：官方文档理解**

首先，回顾内置函数 `__import__` 的定义：

```
__import__(name, globals=None, locals=None, fromlist=(), level=0)
```

在标准实现中，locals 参数被忽略。import 语法糖与 `__import__` 内置函数的对应关系为：

官方文档的三个例子

```python
import spam
spam = __import__('spam', globals(), locals(), [], 0)
```

```python
import spam.ham
spam = __import__('spam.ham', globals(), locals(), [], 0)
```

```python
from spam.ham import eggs, sausage as saus
_temp = __import__('spam.ham', globals(), locals(), ['eggs', 'sausage'], 0)
eggs = _temp.eggs
saus = _temp.sausage
```

晦涩难懂，之后再补充。



Python 导包的常用方法有：import 语句、`__import__` 内置函数、`importlib` 模块。本质上讲，第一种方法实际上会调用第二种方法，而第三种方法会绕过第二种方法，一般而言不推荐直接使用第二种方法。

import 语句与 `__import__` 内置函数的对应关系可以参见[官方文档](https://docs.python.org/zh-cn/3/library/functions.html#__import__)。

怎样完全删除一个已经被导入的包，似乎做不到，参考[链接](https://izziswift.com/unload-a-module-in-python/)

怎样实现自动检测包被修改过或未被导入过，自动进行 reload 操作：待研究

一些疑难杂症：

**实例1：**

```
pkg1
- inference.py  # Detect
pkg2
- inference.py  # Alignment
```

想获得两个包中的模型实例，将两个模型串联进行推断

```python
# 第三个参数是为了防止模型用torch.save(model)的方式保存, 需要额外引入一些包
def get_model_instance(extern_paths, module_cls_pair, extern_import_modules=None, *args, **kwargs):
    sys.path = extern_paths + sys.path
    extern_import_modules = extern_import_modules if extern_import_modules else []
    extern_list = [importlib.import_module(extern_name) for extern_name in extern_import_modules]
    modname, clsname = module_cls_pair
    mod = importlib.import_module(modname)
    instance = getattr(mod, clsname)(*args, **kwargs)
    # 对sys.modules操作可能不够, 未必能删干净
    for extern in extern_import_modules:
        sys.modules.pop(extern)
    sys.modules.pop(modname)
    sys.path = sys.path[len(extern_paths):]
    return instance
```

```
detector = get_model_instance(["pkg1"], ("inference", "Detect"), [])
detector = get_model_instance(["pkg2"], ("inference", "Alignment"), [])
```

```python
detector = get_model_instance(["./detect/facexzoo"], ("inference", "Detect"), ["models"])
```

用于替代

```python
sys.path = ["./detect/facexzoo"] + sys.path
from inference import Detect
import models
sys.path = sys.path[1:]
detector = Detect()
sys.modules.pop("models")
sys.modules.pop("Detect")
```

**实例2：**

假定目录结构为：

```
ROOT
  - models.py
  - load_detr.py
```

文件内容如下：

```python
# models.py
a = 1

# load_detr.py
import torch
model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=False)
from models import a
print(a)
```

运行：

```bash
python load_detr.py
```

报错：

```
ImportError: cannot import name 'a' from 'models'
```

原因在于 `torch.hub.load` 的内部逻辑为：

- 按照 `facebookresearch/detr:main` 去 GitHub 下载原始仓库（https://github.com/facebookresearch/detr）的代码至 `~/.cache/torch/hub` 下。

  备注：此处的 `main` 代表 `main` 分支，代码下载解压完毕后，`~/.cache/torch/hub` 目录下会生成子目录 `facebookresearch_detr_main` 存放当前分支下的代码

  备注：如果原始 GitHub 仓库进行了更新，而本地之前已经下载了之前版本的仓库，可以使用如下方法重新下载

  ```python
  model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=False, force_reload=True)
  ```

- 接下来使用动态 import 的方式，增加了 `~/.cache/torch/hub/facebookresearch_detr_main` 到 sys.path 并使用 importlib 中的相关函数导入代码仓库顶级目录中的 `hubconf.py` 文件里的 `detr_resnet50` 函数，构建模型并下载权重。随后在 sys.path 中移除了 `~/.cache/torch/hub/facebookresearch_detr_main` 路径。 

问题出现在上述仓库的 `hubconf.py` 文件里有这种 import 语句：

```python
from models.backbone import Backbone, Joiner
from models.detr import DETR, PostProcess
def detr_resnet50(...)
```

导致当前目录下的 models 无法被重新导入

修改策略（未必万无一失）：

```python
import torch
model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=False)
import sys
sys.modules.pop("models")
from models import a
```

### 10. Python buid-in fuction and operation

参考资料：[Python 标准库官方文档](https://docs.python.org/3/library/functions.html)

**Truth Value Testing**

任何对象都可以进行 Truth Value Testing（真值测试），即用于 `bool(x)` 或 `if` 或 `while` 语句，具体测试流程为，首先查找该对象是否有 `__bool__` 方法，若存在，则返回 `bool(x)` 的结果。然后再查找是否有 `__len__` 方法，若存在，则返回 `len(x)!=0` 的结果。若上述两个方法都不存在，则返回 `True`。

备注：`__bool__` 方法应返回 `True` 或者 `False`，`__len__` 方法应返回大于等于 0 的整数。若不遵循这些约定，那么在使用 `bool(x)` 与 `len(x)` 时会报错。相当于：

```python
def len(x):
	length = x.__len__()
	check_non_negative_int(length)  # 非负整数检验
	return length
def bool(x):
    if check_bool_exist(x):  # 检查__bool__是否存在
        temp = x.__bool__()
        check_valid_bool(temp)  # bool值检验
        return temp
    if check_len_exist(x):  # 检查__len__是否存在
        return len(x) != 0
    return True
```

备注：`__len__` 只有被定义了之后，`len` 方法才可以使用，否则会报错

**boolean operation: or, and, not**

运算优先级：`非bool运算 > not > and > or`，所以 `not a == b ` 等价于 `not (a == b)`

注意这三个运算符的准确含义如下：

```python
not bool(a)  # not a
a and b  # a if bool(a)==False else b
a or b  # a if bool(a)==True else b
```

```python
12 and 13  # 13
23 or False  # 23
```

**delattr function and del operation**

```python
delattr(x, "foo")  # 等价于 del x.foo
```

### 11. Python 内存管理与垃圾回收（待补充）

### 12. 怎么运行 Python 脚本（感觉没啥有价值的，考虑移除）

主要参考（翻译）自：[RealPython](https://realpython.com/run-python-scripts/)

主要有：

- python xx/yy.py
- python -m xx.yy
- import
- runpy
- importlib
- exec

## python代码打包

### 项目组织形式

参考 [stackoverflow](https://stackoverflow.com/questions/193161/what-is-the-best-project-structure-for-a-python-application)，推荐以类似这种形式组织，注意这些 `__init__.py` 文件是必须的，以确保

```
Project/
|-- bin/
|   |-- project
|
|-- project/
|   |-- test/
|   |   |-- __init__.py
|   |   |-- test_main.py
|   |   
|   |-- __init__.py
|   |-- main.py
|
|-- setup.py
|-- README
```

安装方式为：

```bash
python setup.py install  # 安装在site-packages目录下
pip install /path/to/Project  # 安装在site-packages目录下
pip install -e /path/to/Project  # 安装在当前目录, 适用于开发阶段, 对项目的修改会直接生效, 做修改后无需重新安装包
```

<font color=red>特别说明</font>：关于测试数据与测试代码文件：以下为个人理解，不一定为最佳实践，测试代码中读取数据时应该要获取完整的路径，可以考虑使用 `__file__` 结合相对路径以获取绝对路径。关于这一点，有如下的一个源码分析案例：

源码分析：参考 [scikit-image](https://github.com/scikit-image/scikit-image) 的源代码

```python
from skimage import data
camera = data.camera()
```

其中，`data.camera` 函数的定义位于 `skimage/data/__init__.py`，它进一步调用了同文件下的 `_load("data/camera.png")`，而 `_load` 函数又调用了同文件下的 `_fetch("data/camera.png")`，而 `_fetch` 函数的关键代码如下：

```python
def _fetch(data_filename):
    resolved_path = osp.join(data_dir, '..', data_filename)  # data_dir为该文件的全局变量, 使用了类似os.path.abspath, __file__ 的方式得到
    return resolved_path
```

例子：

项目

```
Foo/
  foo/
  	main.py
  	__init__.py
  	data/
  	  data.txt
  setup.py
```

`foo/main.py`

```python
import os
cur_dir = os.path.dirname(__file__)
with open(os.path.join(cur_dir, "./data/data.txt")) as f:
  print(f.readlines())
```

`foo/__init__.py` 内容为空

`data/data.txt`

```
hello
```

`setup.py`

```python
from setuptools import setup, find_packages

setup(
    name="Foo",
    version="1.0",
    author="yourname",
    packages=find_packages(),
    install_requires=[],
    include_package_data=True,
    package_data={"foo": ["data/*"]}
)
```

安装与使用

```python
# python setup.py install path/to/Foo
from foo import main
```

备注：安装在 `site-packages` 目录下

```
Foo-1.0.dist-info  # Foo与setup.py中的name相对应
foo  # python源代码
```

### 安装依赖包

**第一步：获取requirements.txt**

**方法一: 只获取必要的包\(推荐使用\)**

```text
pip install pipreqs
cd project_path
pipreqs ./ --encoding=utf8
```

**方法二: 获取当前环境下的所有包**

此方案尽量避免使用, 或者在一个干净的虚拟环境下使用

```text
pip freeze > requirements.txt
```

**第二步：利用requirements.txt安装依赖包**

```text
pip install -r requirements.txt
```

### 项目打包详解

问题引出:

* 想开发一个python包上传到PyPI
* 在一个项目中想使用另一个项目的功能: [stackoverflow的一个问题](https://stackoverflow.com/questions/14509192/how-to-import-functions-from-other-projects-in-python)

一些历史, 关于`distutils`, `distutils2`, `setuptools`等, [参考链接](https://zhuanlan.zhihu.com/p/276461821). 大体来说, `distutils`是最原始的打包工具, 是Python标准库的一部分. 而`setuptools`是一个第三方库, 在`setuptools`的变迁过程中, 曾出现过一个分支`distribute`, 现在已经合并回`setuptools`, 而`distutils2`希望充分利用前述三者:`distutils`, `setuptools`, `distribute`的优点成为标准库的一部分, 但没有成功, 并且已经不再维护了. 总之, `distutils`是标准库, `setuptools`是开发者常用的第三方库, 安装好后还额外带着一个叫`easy_install`的第三方管理工具, 而`easy_install`目前用的比较少, `pip`是其改进版. 顺带提一句: python源码安装一般是下载一个压缩包\(先解压, 再编译, 再安装\), 二进制安装一般是下载一个`.egg`或者`.whl`的二进制文件进行安装, 后者已经取代前者成为现今的通用标准. 下面仅介绍基于`setuptools`的使用, 其关键在于编写`setup.py`. 上传到PyPI的方法参考[python官方文档.](https://packaging.python.org/tutorials/packaging-projects/)

**setup.py编写**

首先尝鲜, 在介绍各个参数的用法\(完整列表参见[官方文档](https://setuptools.readthedocs.io/en/latest/references/keywords.html)\)

```text
funniest/
    funniest/
        __init__.py
        text.py
    setup.py
```

```python
from setuptools import setup

setup(name='funniest',  # 包的名称, 决定了用pip install xxx
      version='0.1.1',  # 版本号
      description='The funniest joke in the world',  # 项目描述
      url='http://github.com/storborg/funniest',  # 项目链接(不重要)
      author='Flying Circus',  # 作者名(不重要)
      author_email='flyingcircus@example.com',  # 作者邮箱(不重要)
      license='MIT',
      packages=['funniest'], # 实际上是内层的funniest, 决定了import xxx
      install_requires=[
          'markdown',
      ])  # 依赖项, 优于手动安装requires.txt里的包的方法
```

```text
# 源码安装只需一行
python setup.py install

# 上传到PyPI也只需一行(实际上有三步: 注册包名, 打包, 上传)
python setup.py register sdist upload
# 上传后就可以直接安装了
pip install funniest

# 打包为whl格式(以后补充)
```

已经弃用的参数:

| 已弃用的参数 | 替代品 | 含义 |
| :--- | :--- | :--- |
| `requires` | `install_requires` | 指定依赖包 |
| `data_files` | `package_data` | 指定哪些数据需要一并安装 |

将非代码文件加入到安装包中，注意：这些非代码文件需要放在某个包（即带有 `__init__.py` 的目录）下

* 使用`MANIFEST.in`文件\(放在与`setup.py`同级目录下\), 并且设置`include_package_data=True`, 可以将非代码文件一起安装.
* `package_data`参数的形式的例子为：`{"package_name":["*.txt", "*.png"]}`

**最佳实践**

目前主流的打包格式为 `whl` 格式（取代 `egg` 格式），发布到 PyPi 的包一般使用下面的命令进行安装

```shell
pip install <packagename>
```

实际过程为按照包名 `<packagename>` 在互联网上搜索相应的 .whl 文件，然后进行安装。因此对于源码安装的最佳实践也沿用上述过程，详述如下：

`setup.py` 文件的 `setup` 函数的参数 `packages` 列表长度最好刚好为 1，此时 `setup.py` 文件的 `setup` 函数的参数 `name` 应与 `packages` 的唯一元素相同，且命名全部用小写与下划线，且尽量不要出现下划线。使用下面两条命令安装

```
python setup.py bdist_wheel  # 打包为一个.whl文件，位于当前文件夹的dist目录下
pip install dist/xxx-1.7.4-py3-none-any.whl
```

在 site-packages 目录下会出现类似于如下两个目录

```
xxx-1.7.4.dist-info
xxx
```

备注：whl 格式实际上是 zip 格式，因此可以进行解压缩查看内容

### 发布到 PyPi

参考资料

- [参考realpython](https://realpython.com/pypi-publish-python-package/#different-ways-of-calling-a-package)

## 不能实例化的类

```python
from typing import List
List[int]()  # 注意报错信息
```

## python dict与OrderedDict

关于python自带的字典数据结构, 实现上大致为\([参考stackoverflow回答](https://stackoverflow.com/questions/327311/how-are-pythons-built-in-dictionaries-implemented)\):

* 哈希表\(开放定址法: 每个位置只存一个元素, 若产生碰撞, 则试探下一个位置是否可以放下\)
* python 3.6以后自带的字典也是有序的了\([dict vs OrderedDict](https://realpython.com/python-ordereddict/)\)

说明: 这里的顺序是按照key被插入的顺序决定的, 举例

## 深复制/浅复制/引用赋值

引用赋值: 两者完全一样, 相当于是别名: `x=[1, 2, 3], y=x` 浅赋值: 第一层为复制, 内部为引用: `list.copy(), y=x[:]` 深复制: 全部复制, `import copy; x=[1, 2]; copy.deepcopy(x)`

[Python 直接赋值、浅拷贝和深度拷贝解析 \| 菜鸟教程 \(runoob.com\)](https://www.runoob.com/w3cnote/python-understanding-dict-copy-shallow-or-deep.html)

## Immutable与Hashable的区别

immutable是指创建后不能修改的对象, hashable是指定义了`__hash__`函数的对象, 默认情况下, 用户自定义的数据类型是hashable的. 所有的immutable对象都是hashable的, 但反过来不一定.

另外还有特殊方法`__eq__`与`__cmp__`也与这个话题相关

## PART 3 模块





## python模块导入\(待整理\)

Note 1: 模块只会被导入一次 \(这意味着: 如果对模块进行了修改, 不能利用再次import的方式使修改后的代码生效\).

```python
# test.py文件内容如下
print("abc")
```

```python
# 假设test.py文件处于当前目录下
>>> import test
abc
# python的处理逻辑是按如下顺序去寻找test模块:
# (1) 如果已存在于sys.modules(列表)中, 则不会再次导入
# (2) built-in modules中是否有test, 如果有, 则将其导入, 并将test添加至sys.modules中.
# (3) 依据sys.path(列表)中的目录查找. 注: 默认情况下, 列表的第0个元素为当前目录.
>>> import test        #不会再次导入
```

Note 2: 包是一种特殊的模块. python 3.3之后, 存在两种类型的包: 常规包与命名空间. 简单来说, 包是一个目录, 常规包的目录下有着`__init__.py`文件, 而命名空间则没有.

```text
parent/
    __init__.py
    one/
        __init__.py
    two/
        __init__.py
    three/
        __init__.py
```

导入 `parent.one` 将隐式地执行 `parent/__init__.py` \(首先执行\) 和 `parent/one/__init__.py`. 后续导入 `parent.two` 或 `parent.three` 则将分别执行 `parent/two/__init__.py` 和 `parent/three/__init__.py`.

Note 3:

包具有属性`__path__` \(列表\) 与`__name__` \(字符串\), 模块只有`__name__`属性.

Note 4:

```python
# 绝对导入能使用两种语法:
# import <>
# from <> import <>
import test        #注意import的内容必须是一个模块而不能是模块内定义的函数
from test import f    #可以import包或者函数等

# 相对导入只能使用:
# from <> import <>

# 假设test1.py与test.py在同一目录下, 在test1.py中
from .test import f
# 注意: 假设在test.py同级目录下打开python交互解释器, 上述import语句会报错. 这是由于__main__的特殊性造成的. 具体细节还待考究, 注意__main__的特殊性.
>>> from .test import f
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ModuleNotFoundError: No module named '__main__.test'; '__main__' is not a package
```

一个有趣的例子:

```python
# test文件内容
print("abc")
def f():
    return 1
```

```python
# 注意两点: (1)不能用这种语法引入函数; (2)执行顺序实际上是先执行test.py文件, 再导入test.f, 此时发现test不是一个包, 报错. 但注意, 虽然test.py文件被执行了, 但test模块并未被导入. 具体原因还有待研究.
>>> import test.f
abc
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ModuleNotFoundError: No module named 'test.f'; 'test' is not a package
```

Note 5: 两种包的区别

## Python常用包列表

**conda**

```text
# 科学计算, 机器学习, 作图, 图像, 自然语言处理, 虚拟环境
conda install numpy pandas scipy scikit-learn matplotlib seaborn scikit-image opencv-python Pillow nltk virtualenv virtualenvwrapper
# jupyter
conda install -c conda-forge jupyterlab
# tensorflow请查看官网说明
# torch请查看官网说明
```

**pip**

```text
# 科学计算, 机器学习, 作图, 图像, 自然语言处理, 虚拟环境, jupyterlab
pip install numpy pandas scipy scikit-learn matplotlib seaborn scikit-image opencv-python Pillow nltk virtualenv virtualenvwrapper jupyterlab
# tensorflow请查看官网说明
# torch请查看官网说明
```

## numpy

```python
idx = np.argpartition(x, k, axis=1) # (m, n) -> (m, n)
x[np.range(x.shape[0]), idx[:, k]]  # (m,) 每行的第k大元素值
```

```python
# numpy保存
np.save("xx.npy", arr)
np.load(open("xx.npy"))
```

## argparse

```python
import argparse
# 若不传某个参数一般情况下为None, 若default被指定, 则为default的值（nargs为"?"时为const的值）
parser = argparse.ArgumentParser()

# --base 1 表示base=1，不传表示base=21
parser.add_argument("--base", type=int, default=21)

#  --op1 表示op1=2，不传表示op1=None，--op1 20 表示op1=20
parser.add_argument("--op1", type=int, nargs="?", const=2)
# nargs取值可以为整数/"?"/"*"/"+", 分别表示传入固定数量的参数，传入0/1个参数，传入0个或多个参数，传入1个或多个参数

# --a 表示a=True，不传表示a=False
parser.add_argument("--a", action="store_true")
# 更一般的，可以自定义一个类继承argparse.Action类，然后将这个自定义类名传入action
```

## os

```python
# 求相对路径
os.path.relpath("./data/a/b/c.txt", "./data/a")  # b/c.txt
os.path.splitext("a/b/c.txt")  # ('a/b/c', '.txt')
os.path.expanduser("~/../a.txt")  # /home/username/../a.txt
# abspath与realpath都不会处理~, realpath会返回软连接指向的位置, abspath只会返回软连接
os.path.abspath("~/a.txt")  # /home/username/temp/~/a.txt
os.path.abspath(os.path.expanduser("~/a.txt"))  # /home/username/a.txt
os.path.realpath(os.path.expanduser("~/a.link"))  # /home/username/a.txt

# 最为标准的用法
os.path.abspath(os.path.realpath(os.path.expanduser("~/a.link")))
```





## inspect

### inspect.signature

返回函数的特征标（即原型或者说是参数名列表）

### inspect.stack

用于返回当前的函数调用栈

## pandas

### pandas的apply系列

apply: DataFrame的方法, 可指定axis，应用于行或列

args用于指定额外参数, 但这些参数对于每行或每列是**相同**的

```text
DataFrame.apply(func, axis=0, broadcast=None, raw=False, reduce=None, result_type=None, args=(), **kwds)
```

applymap: DataFrame的方法, 应用于每一个元素

```text

```

| 方法名 | 原型 | 说明 |
| :--- | :--- | :--- |
| `applymap` | `DataFrame.applymap(self, func)` | 逐元素操作 |
| `apply` | `DataFrame.apply(self, func, axis=0, raw=False, result_type=None, args=(), **kwds)` | 按行或列操作 |
| `apply` | `Series.apply(self, func, convert_dtype=True, args(),**kwds)` | 逐元素操作 |
| `map` | `Series.map(self, arg, na_action=None)` | 替换或者逐元素操作 |

### pandas读写excel文件

[参考链接1](https://pythonbasics.org/read-excel/), [参考链接2](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.ExcelWriter.html)

pandas读写excel依赖xlrd, xlwt包, \(ps: 可以尝试直接使用这两个包直接进行读写excel文件\)

```python
df1 = pd.DataFrame({"A": [1, 2, 3]})
df2 = pd.DataFrame({"B": [2, 0, 3]})
df3 = pd.DataFrame({"C": [3, 2, 3]})
with pd.ExcelWriter("path_to_file.xlsx", engine="openpyxl") as writer:
    df1.to_excel(writer, sheet_name='Sheet1')
    df2.to_excel(writer, sheet_name="页2")
with pd.ExcelWriter("path_to_file.xlsx", engine="openpyxl", mode="a") as writer:
    df3.to_excel(writer, sheet_name="Sheet3", index=False)
test = pd.read_excel("path_to_file.xlsx", sheet_name=[0, "Sheet3"])
print(type(test))  # <class 'dict'>
print(test.keys())  # dict_keys([0, 'Sheet3'])
```

直接使用xlrd包示例: [参考链接](https://www.codespeedy.com/reading-an-excel-sheet-using-xlrd-module-in-python/)

```python
# 直接使用xlrd包
import xlrd
wb = xlrd.open_workbook("path_to_file.xlsx")
sheet = wb.sheet_by_index(0)
sheet = wb.sheet_by_name("Sheet3")
# <class 'xlrd.book.Book'> <class 'xlrd.sheet.Sheet'> 4 1
print(type(wb), type(sheet), sheet.nrows, sheet.ncols)
for i in range(sheet.nrows):
    for j in range(sheet.ncols):
        print(sheet.cell_value(i, j), end=" ")
    print()
```

直接使用xlwt包示例: [参考链接](https://www.codespeedy.com/reading-an-excel-sheet-using-xlrd-module-in-python/)

```python
# Writing to an excel sheet using Python 3.x. or earlier 
import xlwt as xw

# Workbook is created 
wb = xw.Workbook() 

# add_sheet is used to create sheet. 
sheet1 = wb.add_sheet('Sheet 1') 
# Specifying style of the elements 
style_value1= xw.easyxf('font: bold 1')
style_value2 = xw.easyxf('font: bold 1, color blue;')
# Input data into rows 
sheet1.write(1, 0, 'Code Speedy', style_value1) 
sheet1.write(2, 0, 'Sarque Ahamed Mollick', style_value2) 

# Input data into columns
sheet1.write(0, 1, 'Position') 
sheet1.write(0, 2, 'No of Posts') 

# 似乎不能写为以.xlsx为后缀的文件(运行不报错, 但使用Excel2019打不开)
wb.save('xlwt codespeedy.xls')  # .xls文件能用Excel2019打开
```

某些情况下`.xlsx`被自动转为了`.xlsm`格式, 可以用pandas进行修复, 注意下面的例子也演示了如何获取一个excel文档的所有sheet名称

```python
x = pd.ExcelFile(r"C:\Users\chenbx\Desktop\调优\默认值.xlsm")
sheet_names = x.sheet_names
y = pd.ExcelWriter(r"C:\Users\chenbx\Desktop\调优\默认值.xlsx")
for sheet in sheet_names:
    df = pd.read_excel(r"C:\Users\chenbx\Desktop\调优\默认值.xlsm", sheet_name=sheet)
    df.to_excel(y, sheet_name=sheet)
y.save()
```

### pandas index相关的操作

```python
# DataFrame.set_index(keys, drop=True, append=False, inplace=False, verify_integrity=False)
df.set_index("key", drop=True)  # 将df["key"]这一列作为新的index, 将原有的index丢弃
df.reset_index(drop=True)  # 将原有的index丢弃, 新的index为默认的[0,1,...], 丢弃的index不作为新列
df.reindex(list_or_index, fill_value=0)  # 只保留list_or_index中的行, 用0填补不存在的行
df.rename(index={1: -1}, columns={"a": "b"}, inplace=False) # 对行或列重命名
```

### merge, join技巧

```python
# 需求: df1与df2, 需要按指定的列名col1, col2做内连接, 希望输出两个dataframe:
# new_df1: 能连上的df1中的部分, 行的相对顺序与df1保持一致, 且列名与df1完全一致
# new_df2: 能连上的df2中的部分, 列名与df2完全一致

# 注:
# 1) 为什么不能用普通的表连接: pandas的dataframe不允许两列的列名相同(实际需求中, df1的列与df2中的列名可能有重复, 并且这些列代表着完全不同的含义)
# 2) col1与col2可以相同, 也可以不同

# 在df1和df2的index都不重要时, 可以使用如下方法
def mymerge(df1, df2, col1, col2):
    df1_new = pd.merge(df1, df2[[col2]].set_index(col2, drop=True), left_on=col1, right_on=col2, how="inner")
    df2_new = pd.merge(df2, df1[[col1]].set_index(col1, drop=True), left_on=col2, right_on=col1, how="inner")
    return df1_new, df2_new
```

## 

## json

```python
import json
dict_json = {'accountID': '123', 'notes': {'sentences color': {1: 50, 0: 50}, 'human color': 2, 'status': '中立'}, 'result': ''}
str_json = json.dumps(dict_json)
print(str_json)
print(json.loads(str_json))
with open("json_format.txt", "w") as fw:
    json.dump(dict_json, fw)
with open("json_format.txt", "r") as fr:
    new_dict = json.load(fr)
print(new_dict)
```

```python
json.dumps(dict)  # dict->str
json.dump(dict, fw)  # dict->write file
json.loads(str)  # str->dict
json.load(fr)  # read file->dict
```

## subprocess

**1. 便捷用法: `subprocess.run`函数**

\[官方文档\]\([https://docs.python.org/3/library/subprocess.html?highlight=subprocess%20run\#subprocess.run](https://docs.python.org/3/library/subprocess.html?highlight=subprocess%20run#subprocess.run)\)

```python
# 函数原型
subprocess.run(args, *, stdin=None, input=None, stdout=None, stderr=None, capture_output=False, shell=False, cwd=None, timeout=None, check=False, encoding=None, errors=None, text=None, env=None, universal_newlines=None, **other_popen_kwargs)
```

```python
import subprocess
cmd = ["dir"]
subprocess.run(cmd)  # 报错: FileNotFoundError: [WinError 2] 系统找不到指定的文件。
subprocess.run(cmd, shell=True)  #正常运行
```

其中`shell`参数的默认值为,`shell=True`表示"命令"\(可能用词不准确\)在shell中执行, 文档中说除非必要, 否则不要设置为True. 注意: 在window下, 上述情况需设置为`True`, 主要原因是windows下`echo`不是一个可执行文件, 而是cmd中的一个命令.

科普\([链接](https://www.cnblogs.com/steamedfish/p/7123749.html)\)

* 一个在windows环境变量PATH目录下的可执行文件\(以`.exe`结尾\), 可以通过`win+R`组合键后敲入文件名进行执行; 而`echo`在windows下不是一个自带的可执行文件, 而是`cmd`窗口中的一个内置命令.
* windows下`cmd`是一个shell, 而平时所说的`dos`是一种操作系统的名字, 而`dos命令`是这个操作系统中的命令. `cmd`窗口下的能执行的命令与`dos`命令有许多重叠之处, 但不能混为一谈.
* 所谓`shell`, 这是一个操作系统中的概念, 不同的操作系统有不同的`shell`, 常见的有: windows下的`cmd`\(命令行shell\), powershell\(命令行shell\), windows terminal\(命令行shell\), 文件资源管理器\(图形化shell\); linux下的bash\(命令行shell, 全称: Bourne Again shell\), shell是一种脚本语言.

## multiprocessing

### Processing

示例1: 低层级API

```python
# 注意观察注释的两行所起的效果
import time
import multiprocessing
def foo(seconds=2):
    print(f"sleep {seconds}s begin.")
    time.sleep(seconds)
    print(f"sleep {seconds}s end.")

t1 = time.time()
p1 = multiprocessing.Process(target=foo, args=(2,))
p2 = multiprocessing.Process(target=foo, args=(2,))
p1.start()
time.sleep(1)
p2.start()
# p1.join()  # 表示停下来等p1运行完
# p2.join()  # 表示停下来等p2运行完
t2 = time.time()
print("运行时间: ", t2 - t1)
```

### Pool

示例2: 高阶API\(常用\)--`multiprocessing.Pool`

```python
import multiprocessing
def func(msg):
    print(multiprocessing.current_process().name + '-' + msg)

pool = multiprocessing.Pool(processes=4) # 创建4个进程
for i in range(10):
    msg = "hello %d" %(i)
    pool.apply_async(func, (msg, ))
pool.close() # 关闭进程池，表示不能在往进程池中添加进程
print(multiprocessing.current_process().name)
pool.join() # 等待进程池中的所有进程执行完毕，必须在close()之后调用
print("Sub-process(es) done.")
```

```text
<<<输出结果>>>
MainProcess
ForkPoolWorker-1-hello 0
ForkPoolWorker-2-hello 1
ForkPoolWorker-1-hello 4
ForkPoolWorker-3-hello 2
ForkPoolWorker-2-hello 5
ForkPoolWorker-1-hello 6
ForkPoolWorker-4-hello 3
ForkPoolWorker-2-hello 7
ForkPoolWorker-1-hello 8
ForkPoolWorker-2-hello 9
Sub-process(es) done.
```

`multiprocessing.Pool`主要有`apply`, `apply_async`, `close`, `imap`, `imap_unordered`, `join`, `map`, `map_async`, `starmap`, `starmap_async`, `terminate`

**close, join, terminate**

`close`指关闭进程池, 即不再往里面添加新的任务

`join`指等待进程池内的所有进程完成任务

`terminate`指立刻中断所有进程的执行

**map, map\_async, starmap, starmap\_async**

首先, python中

```python
list(map(lambda x, y: x+y, [1, 2, 3], [1, 2, 3]))
# 输出结果: [2, 4, 6]

list(itertools.starmap(lambda x, y: x+y, [(1, 1), (2, 2), (3, 3)]))
# 输出结果: [2, 4, 6]
```

`map`只接收单参数的函数, `starmap`是接受多个参数的版本. 这四个函数实际上都调用了`_map_async`, 具体参见源码, `map`会阻塞主进程的执行, 但子进程是并行化的. 在[python官方文档](https://docs.python.org/zh-cn/3.7/library/multiprocessing.html#multiprocessing.pool.Pool.map)中提到, 对于序列很长的时候, 可以使用`imap`并指定`chunksize`参数, 能极大提升效率

```python
# `map`与`map_async`的区别
def foo2(t):
    time.sleep(random.random()*5)
    print(f"got t[0]: {t[0]}, t[1]: {t[1]}, pid: {os.getpid()}, name: {multiprocessing.current_process().name}")
    return t[0] + (t[1] - 5)

def test_map():
    result = pool.map(foo2, ls)
    # 主进程还要干别的事
    time.sleep(7)
    print(result)

def test_map_async():
    result = pool.map_async(foo2, ls)
    # 主进程还要干别的事
    time.sleep(7)
    pool.close()
    pool.join()
    print(result.get())

if __name__ == "__main__":
    start = time.time()
    pool = multiprocessing.Pool(3)
    # print(pool._pool)
    ls = [(i, i) for i in range(10)]
    ls1 = list(range(10))
    ls2 = list(range(10))
    test_map()  # 17s左右
    # test_map_async()  # 9秒左右
    end = time.time()
    print(end-start)
```

**map, apply, imap, imap\_unodered**

|  | multi-args | Concurrence | Blocking | Ordered-results |
| :--- | :--- | :--- | :--- | :--- |
| map | no | yes | yes | yes |
| map\_async | no | yes | no | yes |
| apply | yes | no | yes | no |
| apply\_async | yes | yes | no | no |

concurrence指的是子进程之间能否并发执行, blocking指的是是否阻塞主进程的执行.

最常用的是`map`, 非特殊情况其他不要尝试, 简单的示例如下, 但推荐使用`with`语句

```python
from multiprocessing import Pool

def f(x):
    return x*x

if __name__ == '__main__':
    with Pool(5) as p:
        print(p.map(f, [1, 2, 3]))
```

```python
import time
import random
import multiprocessing
random.seed(1)
# seconds = [random.random()*5 for i in range(10)]
# print(f"seconds: {[round(_, 2) for _ in seconds]}")
# print(f"total time: {sum(seconds): .2f}")
# seconds: [0.67, 4.24, 3.82, 1.28, 2.48, 2.25, 3.26, 3.94, 0.47, 0.14]
# total time:  22.54

def print_func(f):
    def wrapper(*args, **kwargs):
        print(f.__name__)
        return f(*args, **kwargs)
    return wrapper

def foo(x, y):
    time.sleep(random.random()*5)
    print(f"got x: {x}, y: {y}, pid: {os.getpid()}, name: {multiprocessing.current_process().name}")
    return x + (y - 5)

def foo2(t):
    time.sleep(random.random()*5)
    print(f"got t[0]: {t[0]}, t[1]: {t[1]}, pid: {os.getpid()}, name: {multiprocessing.current_process().name}")
    return t[0] + (t[1] - 5)

def bar(e):
    print("Some error happend!")
    print(e)
    return 1000

@print_func
def test_map():
    result = pool.map(foo2, ls)
    print(result)

@print_func
def test_map_async():
    result = pool.map_async(foo2, ls, callback=bar)
    pool.close()
    pool.join()
    print(result.get())

@print_func
def test_apply_async():
    result = []
    for _ in ls:
        result.append(pool.apply_async(foo, _))
    pool.close()
    pool.join()
    result = [_.get() for _ in result]
    print(result)

@print_func
def test_apply():
    result = []
    for _ in ls:
        result.append(pool.apply(foo, _))
    pool.close()
    pool.join()
    print(result)


if __name__ == "__main__":
    start = time.time()
    pool = multiprocessing.Pool(3)
    # print(pool._pool)
    ls = [(i, i) for i in range(10)]
    ls1 = list(range(10))
    ls2 = list(range(10))
    # test_map()
    test_map_async()
    # test_apply()
    # test_apply_async()
    end = time.time()
    print(end-start)
```

### Queue、Pipe

其他一些与进程, cpu相关的测试代码:

```python
import time
import os
import multiprocessing

# 片段1
def foo(seconds=2):
    print(f"sleep {seconds}s begin.")
    time.sleep(seconds)
    print(f"sleep {seconds}s end.")

t1 = time.time()
p1 = multiprocessing.Process(target=foo, args=(2,))
p2 = multiprocessing.Process(target=foo, args=(2,))
p1.start()
time.sleep(1)
p2.start()
p1.join()
p2.join()
t2 = time.time()
print("运行时间: ", t2 - t1)

# 片段2
def func(msg):
    print(multiprocessing.current_process().name + '-' + msg)

pool = multiprocessing.Pool(processes=4) # 创建4个进程
for i in range(6):
    msg = "hello %d" %(i)
    pool.apply_async(func, (msg, ))
pool.close() # 关闭进程池，表示不能在往进程池中添加进程
print(multiprocessing.current_process().name)
pool.join() # 等待进程池中的所有进程执行完毕，必须在close()之后调用
print("Sub-process(es) done.")

# 片段3
print('Process (%s) start...' % os.getpid())
# Only works on Unix/Linux/Mac:
pid = os.fork()
print(pid)
if pid == 0:
    print('I am child process (%s) and my parent is %s.' % (os.getpid(), os.getppid()))
    time.sleep(3)
else:
    print('I (%s) just created a child process (%s).' % (os.getpid(), pid))
    time.sleep(3)

# 片段4
from multiprocessing import Pool
import os, time, random

def long_time_task(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    start = time.time()
    time.sleep(random.random() * 3)
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (name, (end - start)))

print('Parent process %s.' % os.getpid())
p = Pool(None)
for i in range(9):
    p.apply_async(long_time_task, args=(i,))
print('Waiting for all subprocesses done...')
p.close()
p.join()
print('All subprocesses done.')

# 片段5
import subprocess

print('$ nslookup www.python.org')
r = subprocess.call(['nslookup', 'www.python.org'])
print('Exit code:', r)

# 片段6
print('$ nslookup')
p = subprocess.Popen(['nslookup'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
p1 = multiprocessing.Process(target=foo, args=(2,))
print(type(p), type(p1))
output, err = p.communicate(b'set q=mx\npython.org\nexit\n')
print(output.decode('utf-8'))
print('Exit code:', p.returncode)

# 片段7
from multiprocessing import Process, Queue
import os, time, random

# 写数据进程执行的代码:
def write(q):
    print('Process to write: %s' % os.getpid())
    for value in ['A', 'B', 'C']:
        print('Put %s to queue...' % value)
        q.put(value)
        time.sleep(random.random())

# 读数据进程执行的代码:
def read(q):
    print('Process to read: %s' % os.getpid())
    while True:
        value = q.get(True)
        print('Get %s from queue.' % value)

if __name__=='__main__':
    # 父进程创建Queue，并传给各个子进程：
    q = Queue()
    pw = Process(target=write, args=(q,))
    pr = Process(target=read, args=(q,))
    # 启动子进程pw，写入:
    pw.start()
    # 启动子进程pr，读取:
    pr.start()
    # 等待pw结束:
    pw.join()
    # pr进程里是死循环，无法等待其结束，只能强行终止:
    pr.terminate()

# 片段8
>>> import psutil
>>> psutil.pids() # 所有进程ID
[3865, 3864, 3863, 3856, 3855, 3853, 3776, ..., 45, 44, 1, 0]
>>> p = psutil.Process(3776) # 获取指定进程ID=3776，其实就是当前Python交互环境
>>> p.name() # 进程名称
'python3.6'
>>> p.exe() # 进程exe路径
'/Users/michael/anaconda3/bin/python3.6'
>>> p.cwd() # 进程工作目录
'/Users/michael'
>>> p.cmdline() # 进程启动的命令行
['python3']
>>> p.ppid() # 父进程ID
3765
>>> p.parent() # 父进程
<psutil.Process(pid=3765, name='bash') at 4503144040>
>>> p.children() # 子进程列表
[]
>>> p.status() # 进程状态
'running'
>>> p.username() # 进程用户名
'michael'
>>> p.create_time() # 进程创建时间
1511052731.120333
>>> p.terminal() # 进程终端
'/dev/ttys002'
>>> p.cpu_times() # 进程使用的CPU时间
pcputimes(user=0.081150144, system=0.053269812, children_user=0.0, children_system=0.0)
>>> p.memory_info() # 进程使用的内存
pmem(rss=8310784, vms=2481725440, pfaults=3207, pageins=18)
>>> p.open_files() # 进程打开的文件
[]
>>> p.connections() # 进程相关网络连接
[]
>>> p.num_threads() # 进程的线程数量
1
>>> p.threads() # 所有线程信息
[pthread(id=1, user_time=0.090318, system_time=0.062736)]
>>> p.environ() # 进程环境变量
{'SHELL': '/bin/bash', 'PATH': '/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:...', 'PWD': '/Users/michael', 'LANG': 'zh_CN.UTF-8', ...}
>>> p.terminate() # 结束进程
Terminated: 15 <-- 自己把自己结束了
```

## os、sys、subprocess

**片段1**

```python
# 目录结构
# dir
#   --subdir
#       --test1.py
# test1.py内容
import os
ABSPATH = os.path.abspath(__file__)
DIRNAME = os.path.dirname(ABSPATH)
print(ABSPATH)  # 与python命令运行时的目录无关
print(DIRNAME)  # 与python命令运行时的目录无关
print(os.getcwd())  # 与python命令运行时的目录相同
```

```text
# shell中输入
cd dir
python subdir/test1.py
python -m subdir.test1
# 以下为输出结果
# ...\dir\subdir\test1.py
# ...\dir\subdir
# ...\dir
cd subdir
python subdir/test1.py
python -m subdir.test1
# 以下为输出结果
# ...\dir\subdir\test1.py
# ...\dir\subdir
# ...\dir\subdir
```

**片段2**

```python
# 目录结构
# dir
#   --subdir
#       --test.py
#   model.py

# test.py内容
import os
import sys
path = os.path.abspath(__file__)
# model_dir = os.path.dirname(os.path.dirname(path))
model_dir = os.path.join(os.path.dirname(path), "..")
# print(model_dir)
# print(os.path.abspath(model_dir))
sys.path.append(model_dir)
import model

# model.py内容
print(1)
```

```text
# shell中输入
cd dir
python subdir/test.py
cd dir/subdir
python test.py
```

由片段1、2可以得出

* `os.getcwd()`与启动目录相同, 而模块本身的`__file__`与当前模块所在磁盘位置相同
* 使用`sys.path`可以临时修改python的搜索路径, 可以与`__file__`结合使用, 保证可以在任意目录启动

**片段3**

建议不要使用`os.system`模块, 使用`subprocess`模块代替其功能, `subprocess`主要有两种使用方法

```text
pipe = subprocess.Popen("dir && dir", shell=True, stdout=subprocess.PIPE).stdout
print(pipe.read().decode("ISO-8859-1"))

subprocess.call("dir >> 1.txt && dir >> 2.txt", shell=True)subprocess.call("dir >> 1.txt && dir >> 2.txt", shell=True)
```

## 语言检测模块

`langdetect`, `langid`等

## python炫技代码段

```python
def int2str(x):
    return tuple(str(i) if isinstance(i, int) else int2str(i) for i in x)
x = ((1, 2), 1, 3)
int2str(x)  # 输出(('1', '2'), '1', '3')

# 一个综合的例子
from functools import wraps
def to_string(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        def int2str(x):
            return tuple([str(i) if isinstance(i, int) else int2str(i) for i in x])
        return int2str(func(*args, **kwargs))
    return wrapper
@to_string
def f():
    """asd"""
    return ((1, 2), 1, 3)
f.__doc__
```

```python
# for else字句, 若正常结束, 则执行else语句
for n in range(2, 8):
    for x in range(2, n):
        if n % x == 0:
            print( n, 'equals', x, '*', n/x)
            break
    else:
        # loop fell through without finding a factor
        print(n, 'is a prime number')
```

```python
# 慎用, 速度很慢!!
df = pd.DataFrame({"A": [1, 2, 3], "B": [0, 1, 2]})
df[["E", "F"]] = df["A"].apply(lambda x: pd.Series((x, x)))
# 快的方式待整理
```

```python
# 用两个列表创建字典的较快方式(似乎快于字典推导式)
x = dict(zip(key, value))
```

## 发送邮件模块

[未仔细校对过](https://blog.csdn.net/tianshishangxin1/article/details/109856352)

## 压缩与解压模块

### zipfile模块

参考链接: [https://www.datacamp.com/community/tutorials/zip-file\#EZWP](https://www.datacamp.com/community/tutorials/zip-file#EZWP)

```python
# 解压文件
import zipfile
zipname = r'D:\work0126\aa.zip'
out_dir = r"C:\work0126"
pswd = '12345'
with zipfile.ZipFile(zipname) as file:
    # password you pass must be in the bytes you converted 'str' into 'bytes'
    file.extractall(path=out_dir, pwd = bytes(pswd, 'utf-8'))
# 打包为zip
```

## pyhanlp

### 安装说明\(1.7.8版本\)

**step 1**

首先安装JPype1==0.7.0\(版本号必须完全一致\)

```text
pip install JPype1-0.7.0-cp37-cp37m-win_amd64.whl
```

**step 2**

接下来安装pyhanlp\(直接去[网站](https://github.com/hankcs/pyhanlp)下载代码[pyhanlp-master.zip](https://github.com/hankcs/pyhanlp/archive/master.zip), 注意项目名为pyhanlp\)

并下载: jar与配置文件[hanlp-1.7.8-release.zip](http://nlp.hankcs.com/download.php?file=jar), 数据文件[data-for-1.7.5.zip](http://nlp.hankcs.com/download.php?file=data)

注意data1.7.5是被1.7.8版本hanlp兼容的\(实际上也没有data1.7.5版本\), 至此原料已经准备齐全

首先将pyhanlp-master.zip解压, 并进入该目录用如下方式安装

```text
python setup.py install
```

接下来进入安装位置例如:

`C:\Users\54120\anaconda3\envs\hanlp_copy\Lib\site-packages\pyhanlp-0.1.66-py3.7.egg\pyhanlp\static`

将`data-for-1.7.5.zip`解压后的`data`文件夹, `hanlp-1.7.8-release.zip`解压后的`hanlp-1.7.8-sources.jar`, `hanlp-1.7.8.jar`, `hanlp.properties`都放入上述目录下, 最终此目录的结构为:

```text
static
|-  data
    |-  dictionary
    |-  model
    |-  test  (后续示例代码可能将数据下载到这个目录)
    |-  README.url
    |-  version.txt  (内容为1.7.5)
|-  hanlp-1.7.8-sources.jar
│-  hanlp-1.7.8.jar
│-  hanlp.properties
│-  hanlp.properties.in
│-  index.html
│-  README.url
│-  __init__.py
```

**step 3**

修改`hanlp.properties`文件的内容

```text
root=C:/Users/54120/anaconda3/envs/hanlp_copy/Lib/site-packages/pyhanlp-0.1.66-py3.7.egg/pyhanlp/static
```

**step 4**

检查, 在命令行输入

```text
hanlp -v
jar  1.7.8-sources: C:\Users\54120\anaconda3\envs\hanlp_copy\lib\site-packages\pyhanlp-0.1.66-py3.7.egg\pyhanlp\static\hanlp-1.7.8-sources.jar
data 1.7.5: C:\Users\54120\anaconda3\envs\hanlp_copy\Lib\site-packages\pyhanlp-0.1.66-py3.7.egg\pyhanlp\static\data
config    : C:\Users\54120\anaconda3\envs\hanlp_copy\lib\site-packages\pyhanlp-0.1.66-py3.7.egg\pyhanlp\static\hanlp.properties
```

另外, python中应该也要确保可以正常导入

```text
python -c "import pyhanlp"
```

**注意**

上述繁琐的过程使得环境迁移时除了拷贝envs还要修改配置文件.

## Python ctypes使用

ctypes的代码运行效率不如cython

### 1. C/C++程序的编译

### 2. 使用ctypes

参考博客: [Python - using C and C++ libraries with ctypes \| Solarian Programmer](https://solarianprogrammer.com/2019/07/18/python-using-c-cpp-libraries-ctypes/)

以下仅为tutorial\(可能理解不准确\)

#### 2.1 调用动态链接库

以一个例子加以说明: [来源](https://book.pythontips.com/en/latest/python_c_extension.html#ctypes)

```c
// adder.c, 将其编译为adder.so
int add_int(int num1, int num2){return num1 + num2;}
float add_float(float num1, float num2){return num1 + num2;}
```

```python
import ctypes
adder = ctypes.CDLL('./adder.so')  # load the shared object file
res_int = adder.add_int(4, 5)  # Find sum of integers
print("Sum of 4 and 5 = " + str(res_int))

# 注意, 以下用法能确保不出错
p_a, p_b = 5.5, 4.1  # Find sum of floats
c_a, c_b = ctypes.c_float(p_a), ctypes.c_float(p_b)
add_float = adder.add_float
add_float.restype = ctypes.c_float
res = add_float(c_a, c_b)
print("Sum of 5.5 and 4.1 = ", str(res))
```

解释: 由于python与c两种语言在数据结构上有着明显的不同, 因此利用ctypes调用C代码时需要进行相应的类型转换: 以上述的`add_float`为例, python中浮点数都是双精度的\(不妨记为`Python-double`\). 而`adder.c`函数参数都是单精度的, 不妨记为`C-float`, 可以将调用过程细化为几步

* python数据类型转换为c数据类型: `p_a` -&gt; `c_a`, `p_b`-&gt;`c_b`
* c代码调用并返回
* python将c代码返回的结果转换为python类型

为了使用`add_float(1.0, 2.0)`这种形式进行调用, 必须将`1.0`转换为适应`c`的数据形式\(`add_float.argtypes`\), 对于返回值, 同样道理, 也应指定返回时`c->python`的转换\(`add_float.restype`\)

```python
add_float = adder.add_float
add_float.argtypes = [ctypes.c_float, ctypes.c_float]
add_float.restype = ctypes.c_float
print(add_float(1.0, 2.0))  # ok
print(add_float(ctypes.c_float(1.0), ctypes.c_float(2.0)))  # ok
```

```python
add_float = adder.add_float
# add_float.argtypes = [ctypes.c_float, ctypes.c_float]
add_float.restype = ctypes.c_float
print(add_float(1.0, 2.0))  # error
print(add_float(ctypes.c_float(1.0), ctypes.c_float(2.0)))  # ok
```

```python
add_float = adder.add_float
add_float.argtypes = [ctypes.c_float, ctypes.c_float]
# add_float.restype = ctypes.c_float
print(add_float(1.0, 2.0))  # error
print(add_float(ctypes.c_float(1.0), ctypes.c_float(2.0)))  # error
```

经过一番探索后发现, 最好还是指定`argtypes`与`restype`, 前者也许可以不指定, 后者必须指定, 并且不要试图理解不指定`restype`时的结果, 大概是`implement dependent`的东西, 似乎也不能直接用`IEEE 754`浮点数表示法进行解释. 大约是: 不指定`restype`时默认是`c_int`, 深入到底层时, 应该是`C`端传回了一些字节, `Python`端将其解读为`int`, 但不能用同样的比特流解读为浮点型? \(不要深究: ctypes用了cpython安装时的一些东西, 注意: cpython是python的一种实现, 也是最普遍的实现, 与cython是不同的东西\)

argtypes与restype的问题参见: [链接1](https://stackoverflow.com/questions/58610333/c-function-called-from-python-via-ctypes-returns-incorrect-value/58611011#58611011), [链接2](https://stackoverflow.com/questions/24377845/ctype-why-specify-argtypes).

#### 2.2 类似于cython的方式

ctypes也支持直接用python代码写C代码? 但效率不如cython

```python
import ctypes
import ctypes.util
from ctypes import c_int, POINTER, CFUNCTYPE, sizeof
# 如果在linux上, 则将"msvcrt"改为"c"即可
path_libc = ctypes.util.find_library("msvcrt")
libc = ctypes.CDLL(path_libc)
IntArray5 = c_int * 5
ia = IntArray5(5, 1, 7, 33, 99)
qsort = libc.qsort
qsort.restype = None

CMPFUNC = CFUNCTYPE(c_int, POINTER(c_int), POINTER(c_int))

def py_cmp_func(a, b):
    print("py_cmp_func", a[0], b[0])
    return a[0] - b[0]

cmp_func = CMPFUNC(py_cmp_func)
qsort(ia, len(ia), sizeof(c_int), cmp_func)  

for i in ia:
    print(i, end=" ")
```

### 3. ctypes官方文档的学习记录

#### 3.1 python类型与c类型的转换

None, int, bytes, \(unicode\) strings是python中能直接作为C函数参数的数据类型, 其中None代表C中的空指针, bytes与strings作为`char *`与`wchar_t *`的指针, int作为C中的`int`类型, 但注意过大的数字传入C时会被截断. 也就是说上面的`restype`与`argtypes`可以不指定仅限于上述几种数据类型.

> `None`, integers, bytes objects and \(unicode\) strings are the only native Python objects that can directly be used as parameters in these function calls. `None` is passed as a C `NULL` pointer, bytes objects and strings are passed as pointer to the memory block that contains their data \(`char *` or `wchar_t *`\). Python integers are passed as the platforms default C `int` type, their value is masked to fit into the C type.

```python
from ctypes import *
# c_int, c_float是可变的, 这种类型可以改变value
i = c_int(42)  # i: c_long(42)
i.value = -99  # i: c_long(-99)
```

```python
# 注意python中的int类型对应于c_void_p
# c_char_p, c_wchar_p, c_void_p这几种类型改变value实际上改变的是其指向的地址
s = "Hello World"
c_s = c_wchar_p(s)  # c_s: c_wchar_p(139966222), c_s.value: "Hello World"
c_s.value = "Here"  # c_s: c_wchar_p(222222222), c_s.value: "Here"
print(s)  # "Hello World"
```

```python
# create_string_buffer=c_buffer=c_string
# create_unicode_buffer

# 如果需要可变的内存块, 则要使用ctypes.create_string_buffer函数, 此函数有多种调用方式
# 如果需要修改, 则对raw属性或者value属性进行修改即可
p = create_string_buffer(3)  # 开辟3字节空间, 并将值初始化为0
print(sizeof(p), repr(p.raw))  # 3, b'\x00\x00\x00'

p = create_string_buffer(b"Hello")  # 开辟6字节空间
# value用于获得去除空字符后的字节流
print(sizeof(p), repr(p.raw), p.value)  # 6, b'Hello\x00', b'Hello'

p = create_string_buffer(b"Hello", 10)
print(sizeof(p), repr(p.raw)) # 10, b'Hello\x00\x00\x00\x00\x00'
```

```text
>>> printf = libc.printf
>>> printf(b"Hello, %s\n", b"World!")
Hello, World!
14
>>> printf(b"Hello, %S\n", "World!")
Hello, World!
14
>>> printf(b"%d bottles of beer\n", 42)
42 bottles of beer
19
>>> printf(b"%f bottles of beer\n", 42.5)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ArgumentError: argument 2: exceptions.TypeError: Don't know how to convert parameter 2
>>> printf(b"An int %d, a double %f\n", 1234, c_double(3.14))  # 似乎不能用c_float
An int 1234, a double 3.140000
31
>>> # 可能C语言中%f只是用来输出双精度浮点数的?
```

### 4. numpy的C接口

### 5. 关于ctypes的一个提问

Should the argtypes always be specific via ctypes

I want to know the mechanism of `ctypes`, I have written a simple function to do this test. The `C` source code is,

```c
// adder.c, compile it using `gcc -shared -fPIC clib.c -o clib.so`
float float_add(float a, float b) {return a + b;}
```

the `Python` source code is,

```python
import ctypes
from ctypes import c_float
dll_file = "adder.so"
clib = ctypes.CDLL(dll_file)
clib.float_add.argtypes = [c_float, c_float]
clib.float_add.restype = c_float
print(clib.float_add(1., 2.))  # ok, result is 3.0
```

I guess that because of the differences of `Python` and `C`, so the data type should be convert correctly. Concretely, when I specific the `clib.float_add.argtypes`, the process of `clib.float_add(1., 2.)` is, firstly, convert `1.` to `c_float(1.)` and convert `2.` to `c_float(2.)`, which are "C compatible", and the doing computation in `C` side, then the result data of `C` side convert to `Python` data according to the `clib.float_add.restype`. Is it right?

So, if I don't specific the `clib.float_add.argtypes` and always do `Python data -> C data` manually like that, is it always right? But should I always specific the `clib.float_add.restype`?

```python
# clib.float_add.argtypes = [c_float, c_float]
clib.float_add.restype = c_float
print(clib.float_add(c_float(1.), c_float(2.)))  # ok, result is 3.0
```

,,,

Besides that, another thing confused me, the `restype`, I have written another `Python` test code

```python
import ctypes
from ctypes import c_float
dll_file = "clib.dll"
clib = ctypes.CDLL(dll_file)
clib.float_add.argtypes = [c_float, c_float]
print(clib.float_add(1., 2.))  # returns
```

Here, I don't specific the `restype`, I know the default is `int`, when I don't specific that, I can't use `IEEE 754` float representation to explain the result. Is it implement dependent?

## spacy

### 下载模型

解决类似如下命令因为网络原因失效的方法:

```text
python -m spacy download en_core_web_sm
```

去[https://github.com/explosion/spacy-models/查看相应的版本号](https://github.com/explosion/spacy-models/查看相应的版本号), 下载类似如下链接的文件

```text
https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.0.0/de_core_news_sm-3.0.0.tar.gz
```

```text
pip install xxx.tar.gz
```

## huggingface transformers

### 基本使用

### 模型下载目录

设置模型下载位置可参见[官网介绍](https://huggingface.co/transformers/installation.html), 摘抄如下:

**Caching models**

This library provides pretrained models that will be downloaded and cached locally. Unless you specify a location with `cache_dir=...` when you use methods like `from_pretrained`, these models will automatically be downloaded in the folder given by the shell environment variable `TRANSFORMERS_CACHE`. The default value for it will be the Hugging Face cache home followed by `/transformers/`. This is \(by order of priority\):

* shell environment variable `HF_HOME`
* shell environment variable `XDG_CACHE_HOME` + `/huggingface/`
* default: `~/.cache/huggingface/`

So if you don’t have any specific environment variable set, the cache directory will be at `~/.cache/huggingface/transformers/`.

**Note:** If you have set a shell environment variable for one of the predecessors of this library \(`PYTORCH_TRANSFORMERS_CACHE` or `PYTORCH_PRETRAINED_BERT_CACHE`\), those will be used if there is no shell environment variable for `TRANSFORMERS_CACHE`.

### 开源模型

发现有英翻中的模型, [开源模型目录](https://huggingface.co/models), 搜索`zh`, \([https://huggingface.co/Helsinki-NLP/opus-mt-en-zh](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh)\)

使用方法:

```python
# 前三行参照模型地址
# https://huggingface.co/Helsinki-NLP/opus-mt-en-zh/tree/main
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")

# 后面三行参照transformers文档
# https://huggingface.co/transformers/task_summary.html#translation
inputs = tokenizer.encode("translate English to German: Hugging Face is a technology company based in New York and Paris", return_tensors="pt")
outputs = model.generate(inputs, max_length=40, num_beams=4, early_stopping=True)
print(tokenizer.decode(outputs[0]))
```

## 读写excel\(xlsxwriter与pandas\)

pandas与xlsxwriter均支持给输出的excel自定义格式.

```python
# 注意workbook指的是一个excel文件, 而worksheet指的是excel文件当中的一个sheet
import xlsxwriter
workbook  = xlsxwriter.Workbook('filename.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write(0, 0, 'Hello Excel')
workbook.close()
```

```python
# 在生成的excel中操作: “条件格式->管理规则”就可以看到这里定义的规则
import pandas as pd
df = pd.DataFrame({'Data': [1, 1, 2, 2, 3, 4, 4, 5, 5, 6]})
writer = pd.ExcelWriter('conditional.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1', index=False)
workbook  = writer.book
worksheet = writer.sheets['Sheet1']
format1 = workbook.add_format({'bg_color': '#FFC7CE', # 粉色
                               'font_color': '#9C0006'})  # 深红色
format2 = workbook.add_format({'bg_color': '#C6EFCE', # 青色
                               'font_color': '#006100'})  # 深绿色
worksheet.conditional_format('A1:A8', {'type': 'formula', 'criteria': '=MOD(ROW(),2)=0', 'format': format1})
worksheet.conditional_format('A1:A8', {'type': 'formula', 'criteria': '=MOD(ROW(),2)=1', 'format': format2})
writer.save()
```

## re

此部分微妙处比较多, 许多符号例如`?`有着多种含义, 需仔细校对

### pattern的写法

参考

示例里pattern的首位两个正斜杠不是正则表达式的一部分, 示例中所谓的匹配实际上对应的是`re.search`方法, 意思是存在子串能符合所定义的模式, 以下是`python`中`re`模块在单行贪婪模式, 更细节的内容参见后面.

| 写法 | 含义 | 备注 | 示例 |  |
| :--- | :--- | :--- | :--- | :--- |
| `^` | 匹配开头 |  | `/^abc/`可以匹配`abcd`, 但不能匹配`dabc` |  |
| `$` | 匹配结尾 |  | `/abd$/`可以匹配`cabc`, 但不能匹配`abcc` |  |
| `.` | 匹配任意单个字符 | 单行模式下不能匹配`\n` | `/a.v/`可以匹配`acv`, 但不可以匹配`av` |  |
| `[...]` | 匹配中括号中任意一个字符 | 大多数特殊字符`^`, `.`, `*`, `(`, `{`均无需使用反斜杠转义, 但存在例外\(其实还不少\), 例如: `[`与`\`需要用反斜杠转义 | `/[.\["]/`表示匹配如下三个字符: `.`, `[`, `"` |  |
| `[^...]` | 匹配不在中括号中的任意一个字符 |  |  |  |
| `*` | 匹配任意个字符 |  |  |  |
| `{m,n}` | 匹配前一个字符`[m,n]`次 |  |  |  |
| `+` | 匹配前一个字符至少1次 |  |  |  |
| `?` | 匹配前一个字符一次或零次 |  |  |  |
| \` | \` | 或 |  |  |

备注:

* 正则表达式中特有的符号`.*+[]^$()|{}?`, 如果需要表达它们自身, 一般需要转义. 另外, 有以下几个常用的转义字符:

  | 转义写法 | 备注 |  |
  | :--- | :--- | :--- |
  | `\b` | 匹配数字/下划线/字母\(沿用计算机语言中_word_的概念\) |  |
  | `\B` |  |  |
  |  |  |  |

* 贪婪匹配与非贪婪匹配

  `*`, `+`, `?`, `{m, n}`, `{m,}`, `{,n}`均为贪婪模式, 表示尽量匹配更长的字符串, 例如`/t*/`匹配`ttT`中的`tt`. 相反地, 在上述符号后添加`?`后变为非贪婪模式, 表示匹配尽量短的匹配字符串. 有一个特别的例子如下:

  ```python
  # 可以认为原字符串被填补为了 '\0t\0T\0a\0' ?
  re.findall("[tT]*?", "tTa")
  # 输出: ['', 't', '', 'T', '', '']
  ```

* `[...]`形式的pattern存在较多特殊情况
  * `/[tT]*/`可以匹配`tTt`, 即不需要是重复地若干个`t`或者若干个`T`. 类似地`/[tT]{2}/`也可以匹配`tT`.
  * 如果需要匹配`\`, 则需要使用反斜杠将其转义, 例如: `/[\\a]/`表示匹配`\`或者`a`.
  * `/[a-z]/`这种表示是前闭后闭区间\(首尾的两个字符都含在匹配范围内\), 并且必须按照Unicode的顺序前小后大\(见后文\).
  * `[`与`]`不需要进行转义, 如果需要匹配`]`那么`]`必须放在开头, 例如`/[]a]/`表示匹配`]`或者`a`.  但当采用非的语义时, `]`应该放在`^`后面的第一个位置, 例如`/[^]a]/`表示不匹配`]`及`a`.
  * 如果需要匹配`-`, 可以将`-`放在开头或结尾, 或者是用`\-`进行转义.
  * `-`与`]`混合的情形, 例子: `/[-]]/`表示匹配`-]`, `/[]-]/`表示匹配`]`或者`-`.
  * `^`字符无需进行转义, 如果需要匹配`^`, 那么要避免将`^`放在第一个位置即可. 例如: `/[1^]/`表示匹配`1`或者`^`.
  * 还有更多规则...
* `[]`, `{}`, `()`的嵌套问题, _**根据目前所知**_, 一般不能多层嵌套, 具体地, `()`内`[]`和`{}`可以正常使用, 这是因为圆括号只是标记一个范围, 而`[]`与`{}`内部不能嵌套三种括号. 例如: `/([tT]he)/`及`/([tT]{2})/`是合法的pattern.
* `/\1/`这种以反斜杠开头加上数字的表示法表示匹配与上一个圆括号块完全相同的字符串, 标号从1开始, `(?:)`这种写法用于取消当前块的标号
* `(?=)`, `(?!)`被称为Lookahead Assertions, 表示匹配字符但不消耗, 例子`/[A-Za-z]{2}(?!c)/`可以匹配`bag`, 最终获取到`ba`

  ```python
  re.findall("[A-Za-z]{2}(?!c)", "bag") # ["ba"]
  ```

  `(?<=)`, `(?<!)`与上两者类似, 但匹配的是前面的字符

### 常用函数

### 更多说明

贪婪/非贪婪, 单行/多行匹配模式

正则表达式也可以用于字节流的匹配

一些骚操作

```python
import re
inputStr = "hello crifan, nihao crifan";
replacedStr = re.sub(r"(hello )(\w+)(, nihao )(\2)", r"\1crifanli\3\4", inputStr)
replacedStr
```

**python raw string**: `python`中的`raw string`不能以奇数个反斜杠作为结尾, 例如`x=r'\'`这种语句解释器会直接报错, 但`x=r'\\'`会被解释器正确地解释为两个反斜杠. 其原因可以参见这个[问答](https://stackoverflow.com/questions/647769/why-cant-pythons-raw-string-literals-end-with-a-single-backslash), 简述如下: 在python中对`raw string`的解释是碰到`\`就将`\`与下一个字符解释为本来的含义, 也就是说`raw string`出现了反斜杠, 后面必须跟着一个字符, 所以`r'\'`中的第二个单引号会被认为是一个字符, 但这样一来就没有表示字符串结尾的单引号了.

```python
# 测试
x = "a  # SyntaxError: EOL while scanning string literal
x = r"\"  # SyntaxError: EOL while scanning string literal
# 如果希望使用raw string但结尾有单个反斜杠的解决方法
x = r"a\cc" + "\\"  # 表示a\cc\
```

**Unicode与UTF-8的关系**: 参考[阮一峰博客](http://www.ruanyifeng.com/blog/2007/10/ascii_unicode_and_utf-8.html), 简单来说, Unicode是一个**字符代码**, 用整数\(目前至多为21bit\)来表示所有的字符对应关系, 而UTF-8是一种具体的**字符编码**. 前者的重点在于将全世界的字符都有一个唯一的对应关系, 全球公认, 而后者的重点在于在保证能编码所有Unicode中规定地字符集的前提下, 利用更巧妙地方式对这种对应关系进行存储, 方便传输.

### 样例

```python
"(.)\1{2}"  # 用于匹配一个字符3次, 例如:AAA, 注意不能使用(.){3}或.{3}
```

### ~~python re模块的实现~~

使用一个东西, 却不明白它的道理, 不高明

### linux下的通配符\(glob\)

参考资料: [阮一峰博客](http://www.ruanyifeng.com/blog/2018/09/bash-wildcards.html)

通配符又叫做 globbing patterns。因为 Unix 早期有一个`/etc/glob`文件保存通配符模板，后来 Bash 内置了这个功能，但是这个名字被保留了下来。通配符早于正则表达式出现，可以看作是原始的正则表达式。它的功能没有正则那么强大灵活，但是胜在简单和方便。

`?`表示单个字符, `*`代表任意数量个字符, `[abc]`表示方括号内任意一个字符, `[a-z]`表示一个连续的范围, `[^abc]`或`[!abc]`或`[^a-c]`或`[!a-c]`表示排除方括号内的字符

`{abc,def}`表示多字符版本的方括号, 匹配任意`abc`或`def`, 中间用逗号隔开, 大括号可以嵌套, 例如`{j{p,pe}g}`表示`jpg`或`jpeg`.

`{11..13}`表示`11`或`12`或`13`, 但如果无法解释时, 例如: `{1a..1c}`则模式会原样保留.

注意点: `*`不能匹配`/`, 所以经常会出现`a/*.pdf`这种写法

## html转pdf的\(pdfkit\)

依赖于[wkhtmltopdf](https://wkhtmltopdf.org/downloads.html), 安装后\(windows上需添加至环境变量\)可以利用pdfkit包进行html到pdf的转换, 实际体验感觉对公式显示的支持不太好.

```python
# pip install pdfkit
import pdfkit
pdfkit.from_url('https://www.jianshu.com','out.pdf')
```

## black\(自动将代码规范化\)

black模块可以自动将代码规范化\(基本按照PEP8规范\), 是一个常用工具

```text
pip install black
black dirty_code.py
```

## easydict/addict/dotmap

这几个包均是对 python 字典这一基本数据类型的封装，使得字典的属性可以使用点来访问，具体用法及区别待补充：

```python
a.b # a["b"]
```

一些开源项目对这些包的使用情况：

- addict：mmcv
- easydict：
- dotmap：[MaskTheFace](https://github.com/aqeelanwar/MaskTheFace/blob/master/utils/read_cfg.py)

## albumentations（待补充）

基于opencv的数据增强包

## natsort

```python
from natsort import natsorted
x = ["1.png", "10.png", "2.png"]
sorted_x = natsorted(x)
# sorted_x: ["1.png", "2.png", "10.png"]
```

## yacs

作者为 faster rcnn 的作者 Ross Girshick，用于解析 yaml 文件
