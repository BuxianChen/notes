# Python Standard Library

官方文档：https://docs.python.org/3/library/index.html

## logging

基本的用法，日志信息打印在终端并且同时保存在文件中（运行程序的过程中文件内容会不断增加，不是运行完后一次性写入）

```python
import logging
logname = "xx.py"
filename = "x.log"
logger = logging.getLogger(logname)  # logname为可选参数

fh = logging.FileHandler(filename, mode="w")
fh.setFormatter(logging.Formatter(fmt="%(asctime)s %(filename): %(levelname): %(message)s"))
fh.setLevel(logging.INFO)
logger.addHandler(fh)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(fmt="%(asctime)s: %(message)s")
logger.addHandler(ch)

logger.setLevel(logging.INFO)
logger.info("xxx")
```

控制输出内容，不同的日志文件写不同的内容

```python
import logging
def get_logger(logger_name,log_file,level=logging.INFO):
	logger = logging.getLogger(logger_name)
	formatter = logging.Formatter('%(asctime)s : %(message)s', "%Y-%m-%d %H:%M:%S")
	fileHandler = logging.FileHandler(log_file, mode='w')
	fileHandler.setFormatter(formatter)

	logger.setLevel(level)
	logger.addHandler(fileHandler)

	return logger

log_file1 = '1.log'
log_file2 = '2.log'
# logger_name确保不相同,才会生成两个实例
log1 = get_logger('log1', log_file1)
log2 = get_logger('log2', log_file2)
log1.error('log1: error')
log2.info('log2: info')
```

备注：要产生不同的logger，要传递不同的logger_name，例如如下情况得到的两个logger会是一样的：

```python
import logging
import sys
import os

def get_logger(filename):
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    logger = logging.getLogger()

    fh = logging.FileHandler(filename, mode="w")
    fh.setFormatter(logging.Formatter(fmt="%(asctime)s: %(message)s"))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt="%(asctime)s: %(message)s"))
    logger.addHandler(ch)

    logger.setLevel(logging.INFO)
    return logger

logger = get_logger("log.txt")
logger.info("x")

logger2 = get_logger("log2.txt")
logger2.info("y")

logger.info("z")

print(id(logger), id(logger2))
```

## argparse

```python
import argparse
# 若不传某个参数一般情况下为None, 若default被指定, 则为default的值（nargs为"?"时为const的值）
parser = argparse.ArgumentParser()

# --base 1 表示base=1，不传表示base=21
parser.add_argument("-b", "--base", type=int, default=21)

#  --op1 表示op1=2，不传表示op1=None，--op1 20 表示op1=20
# 当nargs指定为"?"时, 默认值用const参数进行指定而非default参数
parser.add_argument("--op1", type=int, nargs="?", const=2)
# nargs取值可以为整数/"?"/"*"/"+", 分别表示传入固定数量的参数，传入0/1个参数，传入0个或多个参数，传入1个或多个参数

# --a 表示a=True，不传表示a=False
parser.add_argument("--a", action="store_true")
# 更一般的，可以自定义一个类继承argparse.Action类，然后将这个自定义类名传入action

# 以下表示--use-a与--use-b至多只能选择一个
group = parser.add_mutually_exclusive_group()
group.add_argument("--use-a", action="store_true", default=False)
group.add_argument("--use-b", action="store_true", default=False)

args = parser.parse_args()
```

备注：parse_args 函数存在 prefix-match的特性, 具体可参考[官方文档](https://docs.python.org/3/library/argparse.html#prefix-matching)的如下例子:

```
>>> parser = argparse.ArgumentParser(prog='PROG')
>>> parser.add_argument('-bacon')
>>> parser.add_argument('-badger')
>>> parser.parse_args('-bac MMM'.split())
Namespace(bacon='MMM', badger=None)
>>> parser.parse_args('-bad WOOD'.split())
Namespace(bacon=None, badger='WOOD')
>>> parser.parse_args('-ba BA'.split())
usage: PROG [-h] [-bacon BACON] [-badger BADGER]
PROG: error: ambiguous option: -ba could match -badger, -bacon
```

有些情况下，可以只解析一部分的命令行参数，而其余参数用其他逻辑进行处理，此时可以使用 `parse_known_args` 函数。备注：`parse_known_args` 函数也适用于 prefix-match 规则。

```python
# test.py
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=True)
args, unknown_args = parser.parse_known_args()
# python test.py -c a.py -d cpu
# args: Namespace(config='a.py'), unknown_args=['-d', 'cpu']
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


## collections

递归定义 `defaultdict`: 参考 (stackoverflow)[https://stackoverflow.com/questions/20428636/how-to-convert-defaultdict-to-dict]

```python
from collections import defaultdict
recurddict = lambda: defaultdict(recurddict)
data = recurddict()
data["hello"] = "world"
data["good"]["day"] = True
```


## typing

typing 模块用于注解

**`Tuple`**

- `Tuple`：元组类型

- `Tuple[int, str]`：第一个元素为整数，第二个元素类型为字符串
- `Tuple[int, ...]`：若干个整数
- `Tuple[Any, ...]`：等价于 `Tuple`

**`Optional`**

`Optional[Sized]` 等同于 `Union[Sized, None]`。

**`Sized`**

`Sized` 表示有一个具有 `__len__` 方法的对象，

```python
from typing import Optional, Sized
def foo(a: Optional[Sized]):
	pass
```

**`Callable`**

- `Callable[[int], str]`：输入是 int 类型，输出是 str 类型的函数
- `Callable[..., str]`：输出是 str 类型的函数，对输入不加约束

**`overload`**

```python
from typing import overload
# 注意: 此处的...是语法
@overload
def foo(name: str) -> str:
	...
@overload
def foo(name: float) -> str:
	...
@overload
def foo(name: int, age: int) -> str:
	...
def foo(name, age=18):
    return "hello" + str(n)
```

**typing.cast**

参考[博客](https://adamj.eu/tech/2021/07/06/python-type-hints-how-to-use-typing-cast/)

```
x = 1
typing.cast(str, x)  # 运行时依旧是整数1, 但mypy检查时认为它是字符串
```

## enum

枚举类型

```python
from enum import Enum
class MyEnum(Enum):
    A = "a"
    B = "b"

# 得到一个枚举类型的实例有几种办法
x = MyEnum.A
y = MyEnum("a")
z = MyEnum["A"]

(x is y) and (y is z)  # True
x == "a"  # False

# 列举所有的取值
for key, value in MyEnum.__members__.items():
    # key分别为["A", "B"]
    # value分别为[MyEnum.A, MyEnum.B]
    ...
```

## dataclasses (python>=3.7 才可用)

引用[官方文档](https://docs.python.org/3/library/dataclasses.html)的说明, 这个模块主要提供了一个针对类的装饰器 `dataclass`，以自动生成一些特殊函数，例如：`__init__`, `__repr__`, `__eq__` 等（对应于 C 语言中的数据结构，即没有实例函数)

> This module provides a decorator and functions for automatically adding generated special methods such as __init__() and __repr__() to user-defined classes

主要的接口为：
- `dataclasses.dataclass`
- `dataclasses.field`

`dataclasses.dataclass` 的简单用法如下：
```python
# 备注: dataclass实际上有很多参数, 例如此处指定fronzen=True, 则初始化后不能再修改数据
@dataclass(frozen=True, eq=True)
class A:
    x: int  # 这里的类型注解是语法强制的, 但运行时不做类型检查
    y: float
    z: str
    def foo(self):
        return self.x * self.y
# A.__init__函数自动生成
a = A(1.0, 2, "x")  # 注意: 此处实际上不会做类型检查
a.foo()
a.x = 3  # 报错
```
备注：`dataclass` 还会自动生成 `__eq__` 等函数, 也可以设定 `eq=False` 抑制这一行为

`field` 的简单用法如下：

```python
@dataclass()
class A:
    # y: list[int] = list()  # 会发生意料不到的情况
    y: list[int] = field(default_factory=list)

a = A()
a.y.append(2)

b = A()
b.y.append(3)

A().y  # 此时会返回[], 但如果不用field函数直接写默认值为list(), 则此时返回为[2, 3]
```

更为深入的细节查阅官方文档：
- 如果 `dataclass` 装饰的类发生继承关系时, 自动生成的 `__init__` 函数的参数顺序一般来说是先父类, 再子类。但还有许多微妙之处
- 将 `dataclass` 中的一些 `field` 仅作为关键字参数如何处理


## unicodedata

unicode 编码的目的是攘括所有的字符，然而它本身也有版本号, python 的每个版本所支持的 unicode 版本号也不相同, 例如: 

- [python 3.8](https://docs.python.org/3.8/library/unicodedata.html) 支持 [UCD version 12.1.0](http://www.unicode.org/Public/12.1.0/ucd)
- [python 3.11](https://docs.python.org/3.11/library/unicodedata.html) 支持 [UCD version 14.0.0](https://www.unicode.org/Public/14.0.0/ucd)


unicode 的完整列表可以参考 [wiki](https://en.wikipedia.org/wiki/List_of_Unicode_characters)，unicode 的字符范围为：`0-0x10FFFF`，因此最多能容纳1114112个码位, 大多数字符的编码范围在 `0-0xFFFF` （最多65536个）之间。一些例子如下：

- `U+0025`: `%`, name 为 `PERCENT SIGN`
- `U+0B90`: `ஐ`, name 为 `TAMIL LETTER AI`

而在编码界，需要区分**编码方式**与**实现方式**，上面所讲的 Unicode 属于**编码方式**的范畴，即规定了字符集。而从实现方式的角度，需要将每个字符映射为一个具体的二进制表示。一个自然的方式是将所有的字符按照 Unicode 的定义方式表示为 6 位 16 进制数，但实际上为了省空间，普遍采用的 Unicode **实现方式**为 `utf-8`、`utf-16` 等，其中最通用的是 `utf-8`，而 `utf-8` 具体的字符与字节的对应关系此处不再做展开。

这里简要举一些 unicodedata 的使用例子，更多复杂的内容请参考维基

```python
import unicodedata
c = "ஐ"
i = ord(c)  # 字符 c 的 unicode 码位, 结果为: 2960 = 0x0B90 = 11*16*16+9*16
c.encode("utf-8")  # 编码为 "utf-8" 时的实际字节表示: b'\xe0\xae\x90'，可以看出utf-8实现中用了3个字节
print("\\u{i:>04x}")  # \\u0b90
unicodedata.name(c)  # 返回字符的名字: "TAMIL LETTER AI"
unicodedata.category(c)  # 字符的类别
unicodedata.normalize('NFC', '\u0043\u0327')  # 使用 NFC 的转换方式将 C 和一个类似逗号的符号合成为 1 个符号，见备注
unicodedata.is_normalized('NFC', '\u0043\u0327')  # False，因为这两个字符可以合并为一个字符，见备注
unicodedata.unidata_version  # unicode data 的版本：'13.0.0', 不同python版本返回值不一样
```

备注：

- 字符类别参见[官网](https://www.compart.com/en/unicode/category)，例如：类别 "Zs" 表示 Space Seperator, 例如空格；但制表符的类别为 "Cc" Control；而中文字符以及很多其他字符被归为 `Lo` Other Letter。
- 字符 normalize，有几种转换方式：NFC、NFD、NFKC、NFKD。例如有些字符有多种表示：U+00C7 (LATIN CAPITAL LETTER C WITH CEDILLA，形状为字母`C`下面有个类似逗号的符号) 也可以被表示为 U+0043 (LATIN CAPITAL LETTER C，字母`C`) U+0327 (COMBINING CEDILLA，类似于一个逗号的符号).


## subprocess (待补充)

- `subprocess` 模块的作用是运行命令行可以执行的命令
- `multiprocessing` 模块的典型用法是用多进程执行同一个python 代码

**subprocess.run**

[官方文档](https://docs.python.org/3/library/subprocess.html?highlight=subprocess%20run#subprocess.run)

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

其中`shell`参数的默认值为,`shell=True`表示"命令"(可能用词不准确)在shell中执行, 文档中说除非必要, 否则不要设置为True. 注意: 在window下, 上述情况需设置为`True`, 主要原因是windows下`echo`不是一个可执行文件, 而是cmd中的一个命令.

科普[链接](https://www.cnblogs.com/steamedfish/p/7123749.html)

* 一个在windows环境变量PATH目录下的可执行文件(以`.exe`结尾), 可以通过`win+R`组合键后敲入文件名进行执行; 而`echo`在windows下不是一个自带的可执行文件, 而是`cmd`窗口中的一个内置命令.
* windows下`cmd`是一个shell, 而平时所说的`dos`是一种操作系统的名字, 而`dos命令`是这个操作系统中的命令. `cmd`窗口下的能执行的命令与`dos`命令有许多重叠之处, 但不能混为一谈.
* 所谓`shell`, 这是一个操作系统中的概念, 不同的操作系统有不同的`shell`, 常见的有: windows下的`cmd`(命令行shell), powershell(命令行shell), windows terminal(命令行shell), 文件资源管理器(图形化shell); linux下的bash(命令行shell, 全称: Bourne Again shell), shell是一种脚本语言.

**subprocess.check_output**

用于执行脚本得到输出结果

```python
import subprocess
output = subprocess.check_output(["echo", "abc"], shell = False)
output.decode()
```

## multiprocessing（待补充）


### multiprocessing.Process

此为一个相对底层的 API，用于直接创建子进程运行 python 代码。

用法一：使用 `multiprocessing.Process`创建进程并传入需要运行的 python 函数及实参

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

方法二：自己写一个类继承 `multiprocessing.Process`，示例如下：

```python
import multiprocessing
class MyProcess(multiprocessing.Process):
    def __init__(self, func, args):
        super().__init__()
        self.func = func
        self.args = args
    
    def run(self):
        self.func(self.args)


def foo(x):
    return x

if __name__ == "__main__":
    process = MyProcess(foo, 1)
    process.start()
    process.join()
```

### Pool

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

**map, map_async, starmap, starmap_async**

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

**map, apply, imap, imap_unodered**

|              | multi-args | Concurrence | Blocking | Ordered-results |
| :----------- | :--------- | :---------- | :------- | :-------------- |
| map          | no         | yes         | yes      | yes             |
| map_async   | no         | yes         | no       | yes             |
| apply        | yes        | no          | yes      | no              |
| apply_async | yes        | yes         | no       | no              |

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


## concurrent.futures

`concurrent.futures`的主要作用是创建异步的线程/进程池。主要的类为 `concurrent.futures.Excutor`(异步调用), `ThreadPoolExecutor`(异步线程池), `ProcessPoolExecutor`(异步进程池)

[concurrent.futures vs multiprocessing.Pool](https://stackoverflow.com/questions/20776189/concurrent-futures-vs-multiprocessing-in-python-3)(待理解)


## 并发执行

python 中涉及关于并发执行的包及主要的类/方法有如下

- **`threading`**: `threading.Thread`, `threading.Pool`, `threading.Queue`
- **`multiprocessing`**: `multiprocessing.Process`, `multiprocessing.Pool`, `multiprocessing.Queue`
- **`concurrent`**: `concurrent.futures.Excutor`, `concurrent.futures.ThreadPoolExecutor`, `concurrent.futures.ProcessPoolExecutor`
- **`subprocess`**: `subprocess.call`
- **`queue`**: `queue.Queue`

此处有多个疑问:

- `concurrent.futures.ProcessPoolExecutor` 与 `multiprocessing.Pool` 之间的区别(待补充)
- `queue.Queue` 与 `multiprocessing.Queue` 之间的区别(待补充)


### 例子1: 使用`concurrent.futures`完成子任务异步调用

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import random
import time
import threading

def foo(uniq_id, data):
    secs = random.random()*2
    time.sleep(secs)
    print(f"{uniq_id}, foo sleep, {secs:.2f}")
    res = {"data_foo": data["data"] + 10}
    return uniq_id, res

def bar(uniq_id, data):
    secs = random.random()
    time.sleep(secs)
    print(f"{uniq_id}, bar sleep, {secs:.2f}")
    res = {"data_bar": data["data"] + 100}
    return uniq_id, res

class Scheduler:
    def __init__(self, names, funcs, pools):
        self.names = names
        self.funcs = funcs
        self.pools = pools
        self.num_executors = len(funcs)


def do_task(uniq_id, data):
    task_results = []
    futures = []
    time.sleep(random.random())
    for i in range(scheduler.num_executors):
        time.sleep(random.random())
        futures.append(scheduler.pools[i].submit(scheduler.funcs[i], uniq_id, data))
    for i in range(scheduler.num_executors):
        time.sleep(random.random())
        task_results.append(futures[i].result()[1])
    result = dict()
    for task_result in task_results:
        result.update(task_result)
    print("处理完毕", uniq_id, data, result)
    return uniq_id, data, result

if __name__ == "__main__":
    scheduler = Scheduler(
        names=["foo", "bar"],
        funcs=[foo, bar],
        pools=[ProcessPoolExecutor(2), ProcessPoolExecutor(2)]
    )


    # 模拟并发请求
    n = 4
    threads = []
    for i in range(n):
        t = threading.Thread(target=do_task, args=(i, {"data": i}))
        t.setDaemon(True)
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()
```

### 例子2: 多个进程修改变量

```python
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
import time
import os

def foo(x, data):
    data.append(x)
    print(x, "Done", id(data), data, os.getpid())

class A:
    def __init__(self):
        self.data = Manager().list()  # 如果是普通的列表, 则self.data将不会被修改
        self.executor = ProcessPoolExecutor(1)
        

    def process(self, i):
        self.executor.submit(foo, i, self.data)  # submit 返回 future 对象

if __name__ == "__main__":
    print(os.getpid())
    a = A()
    a.process(1)
    a.process(2)
    time.sleep(0.2)  # 这里不严谨, 应该用 future 对象, 确保执行完毕
    print(a.data)
```


## re、glob

此部分微妙处比较多, 许多符号例如`?`有着多种含义, 需仔细校对

### pattern的写法

参考

示例里pattern的首位两个正斜杠不是正则表达式的一部分, 示例中所谓的匹配实际上对应的是`re.search`方法, 意思是存在子串能符合所定义的模式, 以下是`python`中`re`模块在单行贪婪模式, 更细节的内容参见后面.

| 写法     | 含义                           | 备注                                                         | 示例                                          |      |
| :------- | :----------------------------- | :----------------------------------------------------------- | :-------------------------------------------- | :--- |
| `^`      | 匹配开头                       |                                                              | `/^abc/`可以匹配`abcd`, 但不能匹配`dabc`      |      |
| `$`      | 匹配结尾                       |                                                              | `/abd$/`可以匹配`cabc`, 但不能匹配`abcc`      |      |
| `.`      | 匹配任意单个字符               | 单行模式下不能匹配`\n`                                       | `/a.v/`可以匹配`acv`, 但不可以匹配`av`        |      |
| `[...]`  | 匹配中括号中任意一个字符       | 大多数特殊字符`^`, `.`, `*`, `(`, `{`均无需使用反斜杠转义, 但存在例外\(其实还不少\), 例如: `[`与`\`需要用反斜杠转义 | `/[.\["]/`表示匹配如下三个字符: `.`, `[`, `"` |      |
| `[^...]` | 匹配不在中括号中的任意一个字符 |                                                              |                                               |      |
| `*`      | 匹配任意个字符                 |                                                              |                                               |      |
| `{m,n}`  | 匹配前一个字符`[m,n]`次        |                                                              |                                               |      |
| `+`      | 匹配前一个字符至少1次          |                                                              |                                               |      |
| `?`      | 匹配前一个字符一次或零次       |                                                              |                                               |      |
| \`       | \`                             | 或                                                           |                                               |      |

备注:

* 正则表达式中特有的符号`.*+[]^$()|{}?`, 如果需要表达它们自身, 一般需要转义. 另外, 有以下几个常用的转义字符:

  | 转义写法 | 备注                                                 |      |
  | :------- | :--------------------------------------------------- | :--- |
  | `\b`     | 匹配数字/下划线/字母\(沿用计算机语言中_word_的概念\) |      |
  | `\B`     |                                                      |      |
  |          |                                                      |      |

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

## ctypes

ctypes的代码运行效率不如cython

### 1. C/C++程序的编译

### 2. 使用ctypes

参考博客: [Python - using C and C++ libraries with ctypes \| Solarian Programmer](https://solarianprogrammer.com/2019/07/18/python-using-c-cpp-libraries-ctypes/)

以下仅为tutorial(可能理解不准确)

#### 2.1 调用动态链接库

以一个例子加以说明: [来源](https://book.pythontips.com/en/latest/python_c_extension.html#ctypes)

```c
// adder.c, 将其编译为adder.so
// gcc -shared -o adder.so -fPIC adder.c
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

解释: 由于python与c两种语言在数据结构上有着明显的不同, 因此利用ctypes调用C代码时需要进行相应的类型转换: 以上述的`add_float`为例, python中浮点数都是双精度的(不妨记为`Python-double`). 而`adder.c`函数参数都是单精度的, 不妨记为`C-float`, 可以将调用过程细化为几步

* python数据类型转换为c数据类型: `p_a` -> `c_a`, `p_b`-> `c_b`
* c代码调用并返回
* python将c代码返回的结果转换为python类型

为了使用`add_float(1.0, 2.0)`这种形式进行调用, 必须将`1.0`转换为适应`c`的数据形式(`add_float.argtypes`), 对于返回值, 同样道理, 也应指定返回时`c->python`的转换(`add_float.restype`)

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

None, int, bytes, (unicode) strings是python中能直接作为C函数参数的数据类型, 其中None代表C中的空指针, bytes与strings作为`char *`与`wchar_t *`的指针, int作为C中的`int`类型, 但注意过大的数字传入C时会被截断. 也就是说上面的`restype`与`argtypes`可以不指定仅限于上述几种数据类型.

> `None`, integers, bytes objects and (unicode) strings are the only native Python objects that can directly be used as parameters in these function calls. `None` is passed as a C `NULL` pointer, bytes objects and strings are passed as pointer to the memory block that contains their data (`char *` or `wchar_t *`). Python integers are passed as the platforms default C `int` type, their value is masked to fit into the C type.

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


## inspect

inspect.signature

返回函数的特征标（即原型或者说是参数名列表）

inspect.stack

用于返回当前的函数调用栈

inspect.isclass(obj)

用于判断 obj 是否为一个类

输出一个实例的完整类名
```bash
import numpy as np
arr = np.array([1, 2])
cls = arr.__class__  # <class numpy.ndarray>
module: str = cls.__module__
name: str = cls.__qualname__  # 优于__name__
```

备注：`__name__` vs `__qualname__`: [stackoverflow](https://stackoverflow.com/questions/58108488/what-is-qualname-in-python)


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

## asyncio

### 更多关于 generator, yield, yield from 的知识

```python
def jumping_range(up_to):
    index = 0
    while index < up_to:
        jump = yield index
        if jump is None:
            jump = 1
        index += jump

if __name__ == '__main__':
    iterator = jumping_range(5)
    print(next(iterator))  # 0
    print(iterator.send(2))  # 2
    print(next(iterator))  # 3
    print(iterator.send(-1))  # 2
    for x in iterator:
        print(x)  # 3, 4
```

执行逻辑为：
- 第一个 `next(iterator)` 会执行到 `yield` 处，返回结果为 `0`
- 接下来的 `send(2)` 会将 `2` 传递给 `jump`，然后再次执行至 `yield` 处，返回结果为 `2`
- ...

备注：
- `send(None)` 等同于 `next`
- 不能去掉第一个 `next` 直接执行 `send(2)`，会报错
