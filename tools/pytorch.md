# Pytorch

pytorch官方文档:

tutorial部分/doc的notes部分

doc的python api部分用于查接口\(部分内容可以作为代码框架的参考\)

doc的Language Binding\(C++部分/Java部分\)

libraries部分包含torchvision等

community部分没探索过

**杂录**

`torch.version.cuda` 变量存储了 cuda 的版本号

`torch.cuda.max_memory_allocated(device=None)`函数用于输出程序从开始运行到目前为止GPU占用的最大内存

`torch.cuda.reset_max_memory_allocated()`函数用于将程序从开始运行到目前为止GPU占用的最大内存设置为0

`torch.nn.Module`的默认模式为**train**, 但为了保险起见, 请手动用`model.train()`与`model.eval()`进行切换.

`torch.nn.Dropout(p)`的行为:

* 测试阶段: 直接将输入原封不动地输出
* 训练阶段: 以`p`的概率将输入的分量置为0, 其余分量变为`1/(1-p)`倍

```python
model = nn.Sequential(nn.Dropout(0.3))
model.train()
model(torch.tensor([1., 2., 3., 4., 5.]))  # [0.0000, 2.8571, 4.2857, 5.7143, 7.1429]

model.eval()
model(torch.tensor([1., 2., 3., 4., 5.]))  # [1., 2., 3., 4., 5.]
```

**finetune\(微调\)**

待补充

**load and save**

```python
class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linear = torch.nn.Linear(3, 3)
    def forward(self, x):
        return self.linear(x)
my_model = MyModule()
torch.save(my_model.state_dict(), "xxx.pth")  # save
my_model.load_state_dict(torch.load("xxx.pth"))  # load

# torchscript
scripted = torch.jit.script(my_model)
scripted.save("yy.pt")
torch.jit.load('yy.pt')
```

**单机多GPU训练**

待补充

**模型量化**

待补充

**torchscript**

大致理解: torchscript将python脚本里写的模型\(即自定义的`torch.nn.Module`子类\)转换为一个中间表示，将这个中间表示保存后，可以使用其他语言或者环境对中间表示进行解析。

主要的方法有两个：`torch.jit.trace`与`torch.jit.script`

```python
# my_model是一个torch.nn.Module子类的对象
traced_model = torch.jit.trace(my_model, input_example)
scripted_model = torch.jit.script(my_model)

# 使用方法与my_model是一样的
traced_model(another_input)
scripted_model(another_input)
```

其中`jit.trace`方法只能追踪`input_example`作为`my_model`的输入时所进行的所有运算过程, 因此如果运算过程中存在分支或循环时, 不能保证得到的`traced_model`与原始的`my_model`完全一致, 而`jit.script`方法是通过分析`my_model`的代码来得到中间表示的，因此可以处理分支或者循环。但这并不代表`jit.script`的功能完全覆盖了`jit.trace`，[参考](https://stackoverflow.com/questions/62626052/what-are-the-differences-between-torch-jit-trace-and-torch-jit-script-in-torchsc)。

`jit.trace`与`jit.script`嵌套，并将模型保存

```python
class MyDecisionGate(torch.nn.Module):
    def forward(self, x):
        return x if x.sum() > 0 else -x

class MyCell(torch.nn.Module):
    def __init__(self, dg):
        super(MyCell, self).__init__()
        self.dg = dg
        self.linear = torch.nn.Linear(4, 4)
    def forward(self, x, h):
        new_h = torch.tanh(self.dg(self.linear(x)) + h)
        return new_h, new_h

x, h = torch.rand(3, 4), torch.rand(3, 4)
# scripted_gate = torch.jit.trace(MyDecisionGate(), x)
scripted_gate = torch.jit.script(MyDecisionGate())
class MyRNNLoop(torch.nn.Module):
    def __init__(self):
        super(MyRNNLoop, self).__init__()
        self.cell = torch.jit.trace(MyCell(scripted_gate), (x, h))
    def forward(self, xs):
        h, y = torch.zeros(3, 4), torch.zeros(3, 4)
        for i in range(xs.size(0)):
            y, h = self.cell(xs[i], h)
        return y, h
rnn_loop = torch.jit.script(MyRNNLoop())
print(rnn_loop.code)
print(rnn_loop.cell.dg.code)

class WrapRNN(torch.nn.Module):
    def __init__(self):
        super(WrapRNN, self).__init__()
        self.loop = torch.jit.script(MyRNNLoop())

    def forward(self, xs):
        y, h = self.loop(xs)
        return torch.relu(y)

traced = torch.jit.trace(WrapRNN(), (torch.rand(10, 3, 4)))
print(traced.code)
print(traced.loop.cell.dg.code)

traced.save('wrapped_rnn.pt')
loaded = torch.jit.load('wrapped_rnn.pt')
```

**C++ Fronted API**

* 一种方案是使用C++ API进行完整的训练与模型保存\(torchscript中定义的格式\)
* 另一种方案是使用Python训练并保存\(必须使用torchscript定义的格式保存\), 然后使用C++ API进行导入

**自动混合精度训练\(Automatic Mixed Precision\)**

docs-&gt;[torch.cuda.amp](https://pytorch.org/docs/stable/amp.html?highlight=torch%20cuda%20amp#module-torch.cuda.amp)

混合精度训练只能在gpu上进行, 因为底层是使用Nvidia为自家gpu提供的`Float16`高效数值运算能力. 平时使用一般只需要用`torch.cuda.amp.GradScaler`以及`torch.cuda.amp.autocast`即可, 并且可以设置`enabled`参数, 当它为`True`时, 则启用amp训练, 否则等价于通常的训练方式. 实际体验上: amp训练计算速度与内存消耗未必快...

**torch.cuda.amp.GradScaler**

**通常用法**

```python
use_amp = True
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

for epoch in range(epochs):
    for input, target in zip(data, targets):
        with torch.cuda.amp.autocast(enabled=use_amp):
            output = net(input)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()
        # scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()
        opt.zero_grad()
```

其中第11行及第12行只有在需要对梯度做修改时才需要做, 此时需要对这两行同时取消注释.

* `scaler.scale(loss).backward()`可以简单地理解为: `(loss*scaler.get_scale()).backward()`, 所以造成的结果是`model.parameters()`中每个参数的`grad`属性被放大了一个倍数\(所有参数共用一个倍数\)
* `scaler.step(optimizer)`可以简单理解为, 首先缩小每个参数的`grad`属性\(原地修改, 并且实际上就是调用了`unscale_`方法\), 之后调用`optimizer.step()`
* `scaler.update()`: 大概是更新放缩比例, 必要性参照下节

**what is it really**

可以通过下面的例子证实上一部分的说明

```python
scaler = torch.cuda.amp.GradScaler()
x = torch.tensor([1., 2.], requires_grad=True, device="cuda:0")
z = torch.tensor([2., 3.], requires_grad=True, device="cuda:0")
y = torch.sum(x) + torch.sum(z)
opt = torch.optim.SGD([x, z], lr=0.001)

scaler.scale(y).backward()
print(x.grad, y)  # [65536, 65536], 8
# opt.step() # 此处若直接用opt调用step, 会利用65536倍的梯度更新, 并且x.grad依然为[65536, 65536], 这种操作会引发错误(相当于学习率被放大), 要避免
scaler.unscale_(opt)
print(x.grad, scaler.get_scale()) # [1, 1], 65535
# torch.nn.utils.clip_grad_norm_([x, z], 1.)  # x.grad = [0.5, 0.5]
torch.nn.utils.clip_grad_norm_(x, 1.)  # 如果将x改为[x, z], 则会将x与z合并起来将梯度剪裁
torch.nn.utils.clip_grad_norm_(z, 1.)
print(x.grad, scaler.get_scale())  # [0.7, 0.7]
scaler.step(opt)
print(x)  # [0.9993, 1.9993]
scaler.update()
```

疑问: 在实现上, 是否取消前一节的两行注释, 代码都能正常工作, 这是怎么做到的呢?

通过阅读源码, 发现`GradScaler`内部用一个参数记录了当前的状态, 如下

```python
class OptState(Enum):
    READY = 0  # __init__以及update函数会将状态更新为READY
    UNSCALED = 1  # 已经调用了unscale_函数
    STEPPED = 2  # 已经调用了step函数
```

当手动调用`unscale_`函数后, 状态会被更新为`UNSCALED`, 而在执行`step`函数时, 如果发现状态为`READY`, 则先调用`unscale_`, 由此做到自动性\(疑问解答\). 另外, `unscale_`函数会首先检查当前状态, 如果是`UNSCALED`或者`STEPPED`直接报错, 因此每次调用`step`后必须使用`update`才能使用`unscale_`

```text
__init__:    -> READY
update:      READY/UNSCALED/STEPPED -> READY
unscale_:    READY -> UNSCALED
step:        READY/UNSCALED -> STEPPED
```

总结: update, unscale\_, step函数的顺序不能乱

