# python-draft

## PART 1 环境配置

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

```text
# 查看pip缓存目录
pip cache dir
# 修改pip缓存目录, 配置文件位置为"C:\\Users\\用户名\\AppData\\Roaming\\pip\\pip.ini"
pip config set global.cache-dir "D:\\Anaconda\\pipDownload\\pip\\cache"
# 用于拷贝环境
pip freeze > requirements.txt
pip install -r requirements.txt
```

### 离线安装python包

有网环境下载安装包

```text
# 下载单个离线包
pip download -d <your_offline_packages_dir> <package_name>
# 批量下载离线包
pip download -d <your_offline_packages_dir> -r requirements.txt
```

将文件拷贝至无网环境安装

```text
# 安装单个离线包
pip install --no-index --find-links=<your_offline_packages_dir> <package_name>
# 批量安装离线包
pip install --no-index --find-links=<your_offline_packages_dir> -r requirements.txt
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

```python
def f(a: int = 1, b: "string" = "") -> str:
    a: int = 1
    b: "str" = "a"
    print(a, b)
a: int = 1
f.__annotations__
```

顺带介绍个骚东西\(上述链接中也有用到\)

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

### 2. 魔术方法与相应的内置函数

#### 2.1 `__str__`与`__repr__`

分别对应于内置方法`str`与`repr`, 一般而言, 前者遵循可读性, 后者遵循准确性. 二者在默认情况\(不重写方法的情况下\)下都会输出类似于`<Classname object at 0x000001EA748D6DC8>`的信息.

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
...             return "__str__"
...
>>> test1 = Test1()
>>> print(test1)  # print使用__str__
__str__
>>> test1
<__main__.Test1 object at 0x000001EA748D6DC8>
```

备注: 在jupyter notebook中, 对`pandas`的`DataFrame`使用`print`方法, 打印出的结果不美观, 但不用`print`却很美观, 原因未知.

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

### 4. 元类

参考资料：[RealPython](https://realpython.com/python-metaclasses/)，[Python 官方文档](https://docs.python.org/3/reference/datamodel.html#metaclasses)，

类是用来构造实例的，因此类也可以被叫做实例工厂；同样地，也有构造类的东西，被称为**元类**。实际上每个类都需要用元类来构造，默认的元类为 `type`。

```python
class A: pass
# 等同于
class A(object, metaclass=type): pass
```

#### `type`

Python 中, type 函数是一个特殊的函数，调用形式有两种：

- `type(obj)`：返回 obj 的类型
- `type()`



`__new__` 函数与 `__init__` 函数

`abc` 模块

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



## python代码打包

### How to import package

[参考realpython](https://realpython.com/pypi-publish-python-package/#different-ways-of-calling-a-package)

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
      version='0.1',  # 版本号
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

## 

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

