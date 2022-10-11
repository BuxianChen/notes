## 运行方法

启用线程将处理完的数据写入kafka, 不影响服务的正常返回

首先启动kafka服务
```
# cd ../mini-demo
docker compose up
```

启动主服务
```
python app.py
```

启动客户端程序
```
python client.py
```

## 关于线程

`thread_helper.py` 中的 `ThreadWithReturnValue` 是 python 线程类 `threading.Thread` 的扩展.

### threading.Thread

默认情况下, python `threading.Thread` 无法得到返回值, 使用方式大致分为两种:

方式1: 使用默认的 `threading.Thread` 传入相关参数

```python
foo = lambda x: print(x)
t = threading.Thread(target=foo, args=(1,))
t.start()
t.join(0.2)  # 主线程等待0.2秒后继续执行
flag = t.is_alive()  # 判断线程是否还在继续执行
```

方式2: 继承 `threading.Thread`, 一般只需要覆盖`__init__`方法及`run`方法即可, 使用时`start` 方法会自动调用 `run` 方法.
```python
class MyWorker(threading.Thread):
    def __init__(self,
                 text="abc",
                 model=None):
        threading.Thread.__init__(self)
        self._text = task
        self._model = model

    def run(self):
        try:
            out = self._model(self._text)
        except Exception:
            out = None
        return out  # 无法接收返回值

t = StatsWorker()
t.start()
t.join(0.2)
flag = t.is_alive()
```


### ThreadWithReturnValue

实现源码如下
```python
# 参考自: https://stackoverflow.com/questions/6893968/how-to-get-the-return-value-from-a-thread-in-python
class ThreadWithReturnValue(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)
    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return
```

`ThreadWithReturnValue` 在实现上继承自 `threading.Thread`, 可以用于接受返回值, 使用时只需要将上述两种方法中的 `threading.Thread` 原地改为 `ThreadWithReturnValue` 即可, 并且可以使用如下方式接收返回值

```python
value = t.join(0.2)  # 若线程没有结束则返回None, 若线程结束则得到返回值
```
