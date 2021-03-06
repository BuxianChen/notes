# Python Packages Usage

## 1. 基础包

### logging

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

### argparse

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


### os

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


### colections

递归定义 `defaultdict`: 参考 (stackoverflow)[https://stackoverflow.com/questions/20428636/how-to-convert-defaultdict-to-dict]

```python
from collections import defaultdict
recurddict = lambda: defaultdict(recurddict)
data = recurddict()
data["hello"] = "world"
data["good"]["day"] = True
```


### typing

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

### subprocess

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

**subprocess.check_output**

用于执行脚本得到输出结果

```python
import subprocess
output = subprocess.check_output(["echo", "abc"], shell = False)
output.decode()
```

### multiprocessing（待补充）

plan：先看懂python官方文档的introduction即可

#### Processing

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

#### Pool

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

|              | multi-args | Concurrence | Blocking | Ordered-results |
| :----------- | :--------- | :---------- | :------- | :-------------- |
| map          | no         | yes         | yes      | yes             |
| map\_async   | no         | yes         | no       | yes             |
| apply        | yes        | no          | yes      | no              |
| apply\_async | yes        | yes         | no       | no              |

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

#### Queue、Pipe

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

### os、sys、subprocess

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

### re、glob

此部分微妙处比较多, 许多符号例如`?`有着多种含义, 需仔细校对

#### pattern的写法

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

#### 常用函数

#### 更多说明

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

#### 样例

```python
"(.)\1{2}"  # 用于匹配一个字符3次, 例如:AAA, 注意不能使用(.){3}或.{3}
```

#### ~~python re模块的实现~~

使用一个东西, 却不明白它的道理, 不高明

#### linux下的通配符\(glob\)

参考资料: [阮一峰博客](http://www.ruanyifeng.com/blog/2018/09/bash-wildcards.html)

通配符又叫做 globbing patterns。因为 Unix 早期有一个`/etc/glob`文件保存通配符模板，后来 Bash 内置了这个功能，但是这个名字被保留了下来。通配符早于正则表达式出现，可以看作是原始的正则表达式。它的功能没有正则那么强大灵活，但是胜在简单和方便。

`?`表示单个字符, `*`代表任意数量个字符, `[abc]`表示方括号内任意一个字符, `[a-z]`表示一个连续的范围, `[^abc]`或`[!abc]`或`[^a-c]`或`[!a-c]`表示排除方括号内的字符

`{abc,def}`表示多字符版本的方括号, 匹配任意`abc`或`def`, 中间用逗号隔开, 大括号可以嵌套, 例如`{j{p,pe}g}`表示`jpg`或`jpeg`.

`{11..13}`表示`11`或`12`或`13`, 但如果无法解释时, 例如: `{1a..1c}`则模式会原样保留.

注意点: `*`不能匹配`/`, 所以经常会出现`a/*.pdf`这种写法

### ctypes

ctypes的代码运行效率不如cython

#### 1. C/C++程序的编译

#### 2. 使用ctypes

参考博客: [Python - using C and C++ libraries with ctypes \| Solarian Programmer](https://solarianprogrammer.com/2019/07/18/python-using-c-cpp-libraries-ctypes/)

以下仅为tutorial\(可能理解不准确\)

##### 2.1 调用动态链接库

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

##### 2.2 类似于cython的方式

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

#### 3. ctypes官方文档的学习记录

##### 3.1 python类型与c类型的转换

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

#### 4. numpy的C接口

#### 5. 关于ctypes的一个提问

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

### inspect

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

## 2. 常用包

### numpy

```python
idx = np.argpartition(x, k, axis=1) # (m, n) -> (m, n)
x[np.range(x.shape[0]), idx[:, k]]  # (m,) 每行的第k大元素值
```

```python
# numpy保存
np.save("xx.npy", arr)
np.load(open("xx.npy"))
```

### pandas

#### pandas的apply系列

apply: DataFrame的方法, 可指定axis，应用于行或列

args用于指定额外参数, 但这些参数对于每行或每列是**相同**的

```text
DataFrame.apply(func, axis=0, broadcast=None, raw=False, reduce=None, result_type=None, args=(), **kwds)
```

applymap: DataFrame的方法, 应用于每一个元素

```text

```

| 方法名     | 原型                                                         | 说明               |
| :--------- | :----------------------------------------------------------- | :----------------- |
| `applymap` | `DataFrame.applymap(self, func)`                             | 逐元素操作         |
| `apply`    | `DataFrame.apply(self, func, axis=0, raw=False, result_type=None, args=(), **kwds)` | 按行或列操作       |
| `apply`    | `Series.apply(self, func, convert_dtype=True, args(),**kwds)` | 逐元素操作         |
| `map`      | `Series.map(self, arg, na_action=None)`                      | 替换或者逐元素操作 |

#### pandas读写excel文件

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

#### pandas index相关的操作

```python
# DataFrame.set_index(keys, drop=True, append=False, inplace=False, verify_integrity=False)
df.set_index("key", drop=True)  # 将df["key"]这一列作为新的index, 将原有的index丢弃
df.reset_index(drop=True)  # 将原有的index丢弃, 新的index为默认的[0,1,...], 丢弃的index不作为新列
df.reindex(list_or_index, fill_value=0)  # 只保留list_or_index中的行, 用0填补不存在的行
df.rename(index={1: -1}, columns={"a": "b"}, inplace=False) # 对行或列重命名
```

#### merge, join技巧

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

### json

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

### easydict/addict/dotmap

这几个包均是对 python 字典这一基本数据类型的封装，使得字典的属性可以使用点来访问，具体用法及区别待补充：

```python
a.b # a["b"]
```

一些开源项目对这些包的使用情况：

- addict：mmcv
- easydict：
- dotmap：[MaskTheFace](https://github.com/aqeelanwar/MaskTheFace/blob/master/utils/read_cfg.py)

## 3. 大杂烩（记录池）

### 语言检测模块

`langdetect`, `langid`等

### 发送邮件模块

[未仔细校对过](https://blog.csdn.net/tianshishangxin1/article/details/109856352)

### 压缩与解压模块

#### zipfile模块

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

### pyhanlp

#### 安装说明\(1.7.8版本\)

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

### spacy

#### 下载模型

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

### huggingface transformers

#### 基本使用

#### 模型下载目录

设置模型下载位置可参见[官网介绍](https://huggingface.co/transformers/installation.html), 摘抄如下:

**Caching models**

This library provides pretrained models that will be downloaded and cached locally. Unless you specify a location with `cache_dir=...` when you use methods like `from_pretrained`, these models will automatically be downloaded in the folder given by the shell environment variable `TRANSFORMERS_CACHE`. The default value for it will be the Hugging Face cache home followed by `/transformers/`. This is \(by order of priority\):

* shell environment variable `HF_HOME`
* shell environment variable `XDG_CACHE_HOME` + `/huggingface/`
* default: `~/.cache/huggingface/`

So if you don’t have any specific environment variable set, the cache directory will be at `~/.cache/huggingface/transformers/`.

**Note:** If you have set a shell environment variable for one of the predecessors of this library \(`PYTORCH_TRANSFORMERS_CACHE` or `PYTORCH_PRETRAINED_BERT_CACHE`\), those will be used if there is no shell environment variable for `TRANSFORMERS_CACHE`.

#### 开源模型

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

#### 离线模型下载实例

`EncoderClassifier` 中有如下注释：

```
classifier = EncoderClassifier.from_hparams(
    ...     source="speechbrain/spkrec-ecapa-voxceleb",
    ...     savedir=tmpdir,
    ... )
```

##### 离线下载模型步骤如下：

```
# 需要先安装git-lfs
git clone https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
# 将hyperparams.yaml中的pretrained_path修改为/home/buxian/Desktop/spkrec-ecapa-voxceleb
```

这样便可以直接使用如下方式导入模型（完全绕过默认路径 `~/.cache/huggingface/hub`）

```
from speechbrain.pretrained import EncoderClassifier
classifier = EncoderClassifier.from_hparams(source="/home/buxian/Desktop/spkrec-ecapa-voxceleb")
```

备注：此处的 git clone 这一方法在离线下载时具有通用性，而修改 `pretrain_path` 是 `speechbrain` 包的内部的逻辑造成的。如果不修改 `pretrain_path`，将无法绕过默认下载路径 `~/.cache/huggingface/hub`。

### 读写excel\(xlsxwriter与pandas\)

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



### html转pdf的\(pdfkit\)

依赖于[wkhtmltopdf](https://wkhtmltopdf.org/downloads.html), 安装后\(windows上需添加至环境变量\)可以利用pdfkit包进行html到pdf的转换, 实际体验感觉对公式显示的支持不太好.

```python
# pip install pdfkit
import pdfkit
pdfkit.from_url('https://www.jianshu.com','out.pdf')
```

### black（自动将代码规范化）

black模块可以自动将代码规范化\(基本按照PEP8规范\), 是一个常用工具

```text
pip install black
black dirty_code.py
```

### albumentations（待补充）

基于opencv的数据增强包

### natsort

```python
from natsort import natsorted
x = ["1.png", "10.png", "2.png"]
sorted_x = natsorted(x)
# sorted_x: ["1.png", "2.png", "10.png"]
```

### yacs

作者为 faster rcnn 的作者 Ross Girshick，用于解析 yaml 文件

### timeout_decorator

超时自动退出装饰器

### python炫技代码段

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

### Python常用包列表

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
