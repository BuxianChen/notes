# Pytorch

pytorch官方文档:

tutorial部分/doc的notes部分

doc的python api部分用于查接口\(部分内容可以作为代码框架的参考\)

doc的Language Binding\(C++部分/Java部分\)

libraries部分包含torchvision等

community部分没探索过

## dataloader

[官方文档](https://pytorch.org/docs/stable/data.html)

```
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
```

对于纯自定义的情形，有如下说明：

一般而言，需要自定义一个继承自 `torch.utils.data.Dataset` 的类，该类必须实现 `__len__`，`__getitem__` 方法。`dataset` 参数为该自定义类的实例。

`sampler`，`batch_size`，`shuffle`，`drop_last` 这组参数与 `batch_sampler` 参数是互斥的关系。`sampler` 参数若指定，只需要是一个定义了 `__len__` 的 `Iterable` 即可（最好是 `torch.utils.data.Sampler` 的子类实例）。`batch_sample` 也类似。

`sampler` 作为迭代器时，每次 `next` 返回的应该是一个下标。而 `batch_sampler` 作为迭代器时，每次 `next` 返回的应该是一个 `batch` 的下标列表。

`collate_fn` 应该是一个 `Callable`，其输入是一个下标列表。

例子：

```python
import torch
import numpy as np

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, length):
        self.length = length
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        return np.array([-idx*2, -idx*2])

class MySampler(torch.utils.data.Sampler):
    def __init__(self, batch_nums):
        self.batch_nums = batch_nums

    def __iter__(self):
        return self.foo()

    def foo(self):
        for i in range(self.batch_nums):
            yield [i, i+1]

    # def __len__(self):
    #     return self.batch_nums

def collect(samples):
    print("passed to collect", samples)
    return samples

dl = torch.utils.data.DataLoader(MyDataset(10),
    batch_sampler=MySampler(20), collate_fn=collect)
for x in dl:
    print(x)
```

## AutoGrad、计算图、自定义算子

### 自定义算子

参考：[pytorch 官方文档](https://pytorch.org/tutorials/intermediate/custom_function_conv_bn_tutorial.html)

例子1：自己实现二维卷积

```python
import torch
from torch.autograd.function import once_differentiable
import torch.nn.functional as F

def convolution_backward(grad_out, X, weight):
    grad_input = F.conv2d(X.transpose(0, 1), grad_out.transpose(0, 1)).transpose(0, 1)
    grad_X = F.conv_transpose2d(grad_out, weight)
    return grad_X, grad_input

class Conv2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight):
        ctx.save_for_backward(X, weight)  # 这是torch.autograd.Function的方法, 保存数据供反向求导使用
        return F.conv2d(X, weight)

    # Use @once_differentiable by default unless we intend to double backward
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        X, weight = ctx.saved_tensors
        return convolution_backward(grad_out, X, weight)

weight = torch.rand(5, 3, 3, 3, requires_grad=True, dtype=torch.double)
X = torch.rand(10, 3, 7, 7, requires_grad=True, dtype=torch.double)
torch.autograd.gradcheck(Conv2D.apply, (X, weight))  # 梯度检查
```

### 梯度惩罚

参考 [amp](https://pytorch.org/docs/stable/notes/amp_examples.html)

```python
for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)

        # Creates gradients
        grad_params = torch.autograd.grad(outputs=loss,
                                          inputs=model.parameters(),
                                          create_graph=True)

        # Computes the penalty term and adds it to the loss
        grad_norm = 0
        for grad in grad_params:
            grad_norm += grad.pow(2).sum()
        grad_norm = grad_norm.sqrt()
        loss = loss + grad_norm

        loss.backward()

        # clip gradients here, if desired

        optimizer.step()
```

此处直接调用了 `torch.autograd.grad` 得到需要计算的梯度，并使用了 `create_graph=True`，这与普通的 `Tensor.backward` 的差异在于：

- `grad_params` 即为求得的所有参数的梯度，但此时所有参数的 `grad` 属性依旧为 `None`。`create_grad=True` 表示为计算梯度所需的算子也将建立计算图
- `loss.backward()` 的作用是求得所有参数的梯度，并更新至它们的 `grad` 属性上，并且销毁整个计算图

从源码上看

```python
# torch/tensor.py
class Tensor:
    def backward(self, gradient=None, retain_graph=None, create_graph=False):
        torch.autograd.backward(self, gradient, retain_graph, create_graph)
# torch/autograd/__init__.py
def backward(
    tensors: _TensorOrTensors,
    grad_tensors: Optional[_TensorOrTensors] = None,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
    grad_variables: Optional[_TensorOrTensors] = None,
    inputs: Optional[_TensorOrTensors] = None,
) -> None:
    if grad_variables is not None:
        warnings.warn("'grad_variables' is deprecated. Use 'grad_tensors' instead.")
        if grad_tensors is None:
            grad_tensors = grad_variables
        else:
            raise RuntimeError("'grad_tensors' and 'grad_variables' (deprecated) "
                               "arguments both passed to backward(). Please only "
                               "use 'grad_tensors'.")
    if inputs is not None and len(inputs) == 0:
        raise RuntimeError("'inputs' argument to backward() cannot be empty.")

    tensors = (tensors,) if isinstance(tensors, torch.Tensor) else tuple(tensors)
    inputs = (inputs,) if isinstance(inputs, torch.Tensor) else \
        tuple(inputs) if inputs is not None else tuple()

    grad_tensors_ = _tensor_or_tensors_to_tuple(grad_tensors, len(tensors))
    grad_tensors_ = _make_grads(tensors, grad_tensors_)
    if retain_graph is None:
        retain_graph = create_graph

    Variable._execution_engine.run_backward(
        tensors, grad_tensors_, retain_graph, create_graph, inputs,
        allow_unreachable=True, accumulate_grad=True)  # allow_unreachable flag
```

```python
# torch/autograd/__init__.py
def grad(
    outputs: _TensorOrTensors,
    inputs: _TensorOrTensors,
    grad_outputs: Optional[_TensorOrTensors] = None,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
    only_inputs: bool = True,
    allow_unused: bool = False
) -> Tuple[torch.Tensor, ...]:
    outputs = (outputs,) if isinstance(outputs, torch.Tensor) else tuple(outputs)
    inputs = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
    overridable_args = outputs + inputs
    if has_torch_function(overridable_args):
        return handle_torch_function(
            grad,
            overridable_args,
            outputs,
            inputs,
            grad_outputs=grad_outputs,
            retain_graph=retain_graph,
            create_graph=create_graph,
            only_inputs=only_inputs,
            allow_unused=allow_unused,
        )

    if not only_inputs:
        warnings.warn("only_inputs argument is deprecated and is ignored now "
                      "(defaults to True). To accumulate gradient for other "
                      "parts of the graph, please use torch.autograd.backward.")

    grad_outputs_ = _tensor_or_tensors_to_tuple(grad_outputs, len(outputs))
    grad_outputs_ = _make_grads(outputs, grad_outputs_)

    if retain_graph is None:
        retain_graph = create_graph

    return Variable._execution_engine.run_backward(
        outputs, grad_outputs_, retain_graph, create_graph,
        inputs, allow_unused, accumulate_grad=False)
```

因此在本质上，`Tensor.backward` 实际上就是 `torch.autograd.backward`，它与 `torch.autograd.grad` 都是调用 `Variable._execution_engine.run_backward` 函数，只不过前者调用方式为：

```python
# 注意这个结果不被返回, 而是直接累积到叶子节点的梯度上
Variable._execution_engine.run_backward(
    tensors, grad_tensors_, retain_graph, create_graph, inputs,
    allow_unreachable=True, accumulate_grad=True)  # allow_unreachable flag
```

后者的调用方式为

```python
# 注意这个结果被返回, 但不累积到叶子节点的梯度上
Variable._execution_engine.run_backward(
    outputs, grad_outputs_, retain_graph, create_graph,
    inputs, allow_unused, accumulate_grad=False)
```

## torch.nn.Module

### finetune

待补充

### load and save

```python
class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linear = torch.nn.Linear(3, 3)
    def forward(self, x):
        return self.linear(x)
# 只保存模型参数
my_model = MyModule()
torch.save(my_model.state_dict(), "xxx.pth")  # save
my_model.load_state_dict(torch.load("xxx.pth"))  # load

# torchscript
scripted = torch.jit.script(my_model)
scripted.save("yy.pt")
torch.jit.load('yy.pt')
```

注记：不推荐使用如下方式保存模型，会出现无法 `load` 的情况，具体原因尚不明确。参见[问答](https://discuss.pytorch.org/t/modulenotfounderror-no-module-named-network/71721)

```python
# 保存模型参数及定义
torch.save(my_model, "model.pth")
my_model = torch.load("model.pth")
```

## optimizer

使用范例如下：（引用自[官方文档](https://pytorch.org/docs/stable/optim.html)），注意官方推荐 `zero_grad()->forward->loss.backward()->optimizer.step()` 的方式进行参数更新, 并且 `scheduler.step` 放在整个 epoch 的最后进行更新

```python
model = [Parameter(torch.randn(2, 2, requires_grad=True))]
optimizer = SGD(model, 0.1)
scheduler = ExponentialLR(optimizer, gamma=0.9)
for epoch in range(20):
    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5., norm_type=2)
        optimizer.step()
    scheduler.step()
```

### torch.optim

```
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam([var1, var2], lr=0.0001)
```

如果模型的各个参数需要使用不同的学习率, 则可以使用如下两种方式

```python
optim.SGD([{'params': model.base.parameters()},
	{'params': model.classifier.parameters(), 'lr': 1e-3}],
	lr=1e-2, momentum=0.9)  # 即设定默认的lr为1e-2, 默认的momentum为0.9
```

```python
# model is a instanse of torch.nn.Module
g0, g1, g2 = [], [], []  # optimizer parameter groups
for v in model.modules():
    if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
        g2.append(v.bias)
    if isinstance(v, nn.BatchNorm2d):
        g0.append(v.weight)
    elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
        g1.append(v.weight)
optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})
optimizer.add_param_group({'params': g2})
```

### torch.optim.lr_scheduler

使用自定义的学习率调整策略可以使用 `LambdaLR` 类

```python
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

x = torch.tensor([1., 2.], requires_grad=True)
optimizer = SGD([x], 0.1)
scheduler = LambdaLR(optimizer, lambda x: (x+1)**2)
# 没有调用 scheduler.step() 之前, 学习率为 0.1 * (0 + 1)**2 = 0.1
for i in range(3):
    optimizer.zero_grad()
    y = torch.sum(x)
    y.backward()
    print(f"before optimizer.step(), x.grad: {x.grad}, x: {x}")
    optimizer.step()
    print(f"after optimizer.step(): x.grad, {x.grad}, x: {x}")
    scheduler.step()
    print(scheduler.get_lr())  # 获取当前的学习率, 注意返回的是列表, 依次为每个参数组的学习率
# 简化版的输出结果:
# before optimizer.step(), x.grad: [1., 1.], x: [1., 2.]
# before optimizer.step(): x.grad, [1., 1.], x: [0.9000, 1.9000]
# [0.4]
# before optimizer.step(), x.grad: [1., 1.], x: [0.9000, 1.9000]
# before optimizer.step(): x.grad, [1., 1.], x: [0.5000, 1.5000]
# [0.9]
# before optimizer.step(), x.grad: [1., 1.], x: [0.5000, 1.5000]
# before optimizer.step(): x.grad, [1., 1.], x: [-0.4000,  0.6000]
# [1.6]
```

对不同的参数设置了不同的学习率及学习率调整策略时, `get_lr` 会返回不同参数组的当前学习率

```python
x = torch.tensor([1., 2.], requires_grad=True)
z = torch.tensor(3., requires_grad=True)
optimizer = SGD([{'params': [x], 'lr': 1}, {'params': [z], 'lr': 2}])
scheduler = LambdaLR(optimizer, [lambda x: (x+1)**2, lambda x: x+1])
print(scheduler.get_lr())
for i in range(2):
    optimizer.zero_grad()
    y = torch.sum(x) + z
    y.backward()
    optimizer.step()
    scheduler.step()
    print(scheduler.get_lr())
# 简化版的输出:
# [1, 2]
# [4, 4]
# [9, 6]
```

### 梯度剪裁

梯度剪裁的用法例子如下

```python
optimizer.zero_grad()
outputs = model(data)
loss = loss_fn(outputs, targets)
nn.utils.clip_grad_norm_(model.parameters(), max_norm=5., norm_type=2)
optimizer.step()
```

```python
# Pytorch 1.9.1 源码(有删减注释1) torch/nn/utils/clip_grad.py
def clip_grad_norm_(
        parameters: _tensor_or_tensors, max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = False) -> torch.Tensor:
    r"""
    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.
    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        norms = [p.grad.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    if total_norm.isnan() or total_norm.isinf():
        if error_if_nonfinite:
            raise RuntimeError("")
        else:
            warnings.warn("", FutureWarning, stacklevel=2)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef.to(p.grad.device))
    return total_norm
```

## 自动混合精度训练\(Automatic Mixed Precision\)

docs-&gt;[torch.cuda.amp](https://pytorch.org/docs/stable/amp.html?highlight=torch%20cuda%20amp#module-torch.cuda.amp)

混合精度训练只能在gpu上进行, 因为底层是使用Nvidia为自家gpu提供的`Float16`高效数值运算能力. 平时使用一般只需要用`torch.cuda.amp.GradScaler`以及`torch.cuda.amp.autocast`即可, 并且可以设置`enabled`参数, 当它为`True`时, 则启用amp训练, 否则等价于通常的训练方式. 实际体验上: amp训练计算速度与内存消耗未必快...

混合精度训练时，模型的参数是 float32 类型的？

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
        optimizer.zero_grad()
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

**原理**

```python
with autocast():
	output = model(input)  # model, input: all float32
    loss = loss_fn(output, target)  # maybe float32 or float16
scaler.scale(loss).backward()  # grad always float32
# unscale_调用torch._amp_non_finite_check_and_unscale_
scaler.unscale_(optimizer)  # 缩小倍数, 并记录是否存在inf/nan的情况
scaler.step(optimizer)  # 若上一步记录发现inf/nan, 则跳过step
scaler.update()  # 根据inf/nan的情况以及迭代次数来更新_scale
```

## cuda and distributed

### 入门

#### 环境变量

```python
# 最佳实践需设定这一项
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
```

设定使用的 GPU，PCI_BUS_ID 是表示 `0,1` 代表的是物理 GPU ID 为 `0,1` 的两块 GPU。[参考](https://www.jianshu.com/p/d10bfee104cc)

```python
import os
import torch
# 这两行顺序不能乱，否则后一行的结果会不正确
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
torch.cuda.device_count()  # 2
```

### 简介

[参考官方文档](https://pytorch.org/tutorials/beginner/dist_overview.html)

### 术语释义

[参考官方文档](https://pytorch.org/docs/1.9.0/notes/cuda.html)

#### TensorFloat-32(TF32) on Ampere devices

pytorch 1.7 之后，可以通过设置这两个参数为 True 提升 32 位浮点运算的速度，默认值即为 True。设置为 True 之后，运算精度会变低许多，但速度会快很多

```python
# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True
```

#### Asynchronous execution

##### CUDA streams

CUDA streams 是 Nvidia CUDA C 官方文档中所用的术语，一个 CUDA stream 表示的是一串在 GPU 上执行的命令。这一串命令将保证按次序执行，但用户可以创建多个 CUDA streams，不同的 CUDA stream 中的指令是并发执行的。在 Pytorch 中，每块 GPU 都有一个默认的 CUDA stream，其特别之处是会在必要的时候自动做同步。但用户也可以自己创建新的 CUDA stream，并将命令放在创建的 CUDA stream 中执行。此时，必要的同步操作需要用户自己做，例如下面的程序没有做好同步，因此计算出的 `B` 是错误的。

```python
cuda = torch.device('cuda')
s = torch.cuda.Stream()  # Create a new stream.
A = torch.ones((32, 64, 448, 448), device=cuda)  # execute in default stream
weight = torch.rand((64, 64, 5, 5), device=cuda)  # execute in default stream
A = torch.conv2d(A, weight, padding=2)  # execute in default stream
A.zero_().normal_().zero_()  # execute in default stream
with torch.cuda.stream(s):
    # sum() may start execution before default stream finishes!
    # torch.cuda.synchronize()  # 加上这行可以避免错误
    B = torch.sum(A)
```

一般可以使用 `torch.cuda.synchronize()` 或 `torch.cuda.Stream.synchronize()` 或 `torch.cuda.Stream.wait_stream(stream)` 等方法进行同步。完整 API 参见[官方文档](https://pytorch.org/docs/stable/generated/torch.cuda.Stream.html)。

### torch.nn.DataParallel

#### 使用

例子 1

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return self.len
rand_loader = DataLoader(dataset=RandomDataset(100, 5), batch_size=30, shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
	print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)
model.to(device)  # 注意此处是设定主GPU
for data in rand_loader:
    input = data.to(device) # 直接将数据放到主GPU上即可
    output = model(input)
    print("Outside: input size", input.size(), "output_size", output.size())
```

从代码上看，实际上只需要增加一行即可

```
model = nn.DataParallel(model)
```

例子 2

```python
import timm
model = timm.create_model("resnet18")  # 此处的是否将模型放在GPU上均可以
model = torch.nn.DataParallel(model, [0, 1])
inp = torch.rand([33, 3, 224, 224]).cuda(2)
model(inp)  # 此处的inp可以任何位置, 包括CPU, 第0, 1块GPU上,甚至是其他GPU上
```

#### 原理

torch 1.9.0 版本关于 DataParallel 的函数原型为：

```python
class torch.nn.DataParallel(module, device_ids=None, output_device=None, dim=0)
```

其中 `dim` 表示输入的数据将会在这一维度被平均分配到各个 GPU 中。

> The parallelized `module` must have its parameters and buffers on `device_ids[0]` before running this [`DataParallel`](https://pytorch.org/docs/master/generated/torch.nn.DataParallel.html#torch.nn.DataParallel) module

### torch.nn.parallel.DistributedDataParallel

#### torch/distributed/launch.py

##### 使用

参考 Github 项目：[分布式训练的例子](https://github.com/tczhangzhi/pytorch-distributed)

[pytorch-src/dist_example.py](pytorch-src/dist_example.py)

##### 原理 & 源码

有时会见到以这种方式启动训练脚本

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr 127.0.0.1 --master_port 29500 train.py
```

参数含义解释如下：

local_rank一般是只每个机器内部的GPU编号
os.environ['WORLD_SIZE']
单机: nnodes=1, 八卡: nproc_per_node=8

torch 1.6.0 中 `torch/distributed/launch.py` 源码（只去除了注释）如下：

```python
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import subprocess
import os
from argparse import ArgumentParser, REMAINDER

def parse_args():
    parser = ArgumentParser()
    # Optional arguments for the launch helper
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--nproc_per_node", type=int, default=1)
    parser.add_argument("--master_addr", default="127.0.0.1")
    parser.add_argument("--master_port", default=29500, type=int)
    parser.add_argument("--use_env", default=False, action="store_true")
    parser.add_argument("-m", "--module", default=False, action="store_true")
    parser.add_argument("--no_python", default=False, action="store_true")
    # positional
    parser.add_argument("training_script", type=str)
    # rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()
""" 备注: 使用下面的命令启动时
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 train.py --lr 0.2 --layers 34
training_srcipt为"train.py", training_script_args为["--lr", "0.2", "--layers", "34"]
"""
def main():
    args = parse_args()
    # world size in terms of number of processes
    dist_world_size = args.nproc_per_node * args.nnodes
    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = args.master_addr
    current_env["MASTER_PORT"] = str(args.master_port)
    current_env["WORLD_SIZE"] = str(dist_world_size)
    processes = []
    if 'OMP_NUM_THREADS' not in os.environ and args.nproc_per_node > 1:
        current_env["OMP_NUM_THREADS"] = str(1)
        print("Setting OMP_NUM_THREADS environment variable for each process "
              "to be {} in default, to avoid your system being overloaded, "
              "please further tune the variable for optimal performance in "
              "your application as needed. \n".format(current_env["OMP_NUM_THREADS"]))
    for local_rank in range(0, args.nproc_per_node):
        # each process's rank
        dist_rank = args.nproc_per_node * args.node_rank + local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)
        # spawn the processes
        with_python = not args.no_python
        cmd = []
        if with_python:
            cmd = [sys.executable, "-u"]  # sys.excutable由sys.argv[0]及若干个环境变量决定
            if args.module:
                cmd.append("-m")
        else:
            if not args.use_env:
                raise ValueError("When using the '--no_python' flag, you must also set the '--use_env' flag.")
            if args.module:
                raise ValueError("Don't use both the '--no_python' flag and the '--module' flag at the same time.")
        cmd.append(args.training_script)
        if not args.use_env:
            cmd.append("--local_rank={}".format(local_rank))
        cmd.extend(args.training_script_args)
        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)
    for process in processes:
        process.wait()  # 等待所有子进程结束
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode,cmd=cmd)
if __name__ == "__main__":
    main()
```



例如：使用下面的命令启动训练脚本 `train.py` 时

```
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 train.py --lr 0.2 --layers 34 -m 7
```

launch 脚本实际做的事情是利用 train.py 之前的参数设定开的进程的环境变量

```
--nnodes
--node_rank
--nproc_per_node
--master_addr
--master_port
--use_env
-m, --module
--no_python
# training_script
# training_script_args
```

备注：由于REMAINDER的特性，training_script_args仍然会包含 `["-m", "7"]`

最终转换为开启多个进程执行

```
XXX=111 YYY=111 python -u train.py --local_rank={k} --lr 0.2 --layers 34 -m 7
```

因此，使用这个脚本时，train.py 必须能解析 --local_rank 这个参数



**原理**

```
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
def step1(x):
    return torch.matmul(w, x)

def step2(x):
    return torch.sum(x)

def end2end(x):
    return step2(step1(x))
```

```
w = torch.tensor([1., 2.], requires_grad=True)
x = torch.tensor([3., 4.])
z = end2end(x)
z.backward()
w.grad
```

```
w = torch.tensor([1., 2.], requires_grad=True)
x = torch.tensor([3., 4.])
y = step1(x)
y1 = y.clone().detach().requires_grad_(True)
z = step2(y1)
z.backward()
y.backward(y1.grad)
w.grad
```

#### torch/distributed/run.py 与 torchrun

在 1.0.0-1.8.1 版本中，均使用 `torch/distributed/launch.py`（简称 `launch.py`）来启动。但在 1.9.0 及 1.9.1 版本中，官方文档中说 `torch/distributed/run.py`（简称 `run.py`）已被弃用，推荐用 `run.py` 来启动，而 1.9.0 与 1.9.1 版本的 `launch.py` 改为了调用 `run.py` 内的相关函数。而在 1.10.0 版本中，`setup.py` 文件引进了一项改动，又变为使用脚本 `torchrun` 进行启动，但实质上与 `run.py` 是一样的。而 1.9.0 版本的 `run.py` 本质上是调用了合并自 `TorchElastic` 的 `torch.distributed.elastic` 子模块下的内容。

??-1.9.1 版本的 `setup.py` 文件关于 `entry_points` 的写法：

```python
entry_points = {
        'console_scripts': [
            'convert-caffe2-to-onnx = caffe2.python.onnx.bin.conversion:caffe2_to_onnx',
            'convert-onnx-to-caffe2 = caffe2.python.onnx.bin.conversion:onnx_to_caffe2',
        ]
    }
```

1.10.0 版本的 `setup.py` 文件关于 `entry_points` 的写法：

```python
entry_points = {
        'console_scripts': [
            'convert-caffe2-to-onnx = caffe2.python.onnx.bin.conversion:caffe2_to_onnx',
            'convert-onnx-to-caffe2 = caffe2.python.onnx.bin.conversion:onnx_to_caffe2',
            'torchrun = torch.distributed.run:main',
        ]
    }
```

总结一下：torch 1.8.1 及之前的 `launch.py` 文件实现逻辑及用法如上一节所述。torch 1.9.0 版本由于 TorchElastic 引入了 torch 中，所以使用了新的 `run.py` 作为启动文件（使用 TorchElastic 的功能），为保持兼容性，`launch.py` 的用法维持原状，但本质上也是在使用 `run.py`。

因此，只需介绍 torch 1.9.1 版本的 `run.py` 即可

### TorchElastic

TorchElastic 原本是一个[独立的包](https://github.com/pytorch/elastic)，但 Pytorch 1.9.0 版本将 TorchElastic 进行了集成，位于 `torch.distributed.elastic` 子模块下。参见上一节关于 `torch/distributed/run.py` 的介绍。

### RPC

DataParallel、DistributedDataParallel、TorchElastic 均属于 DataPallel，RPC 为模型并行

### c10d

c10d 是两大类并行方法的共同底层依赖（通信机制）

## 模型量化

待补充

## torchscript

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
class MyDecisionGate(torch.nn.Module):    def forward(self, x):        return x if x.sum() > 0 else -xclass MyCell(torch.nn.Module):    def __init__(self, dg):        super(MyCell, self).__init__()        self.dg = dg        self.linear = torch.nn.Linear(4, 4)    def forward(self, x, h):        new_h = torch.tanh(self.dg(self.linear(x)) + h)        return new_h, new_hx, h = torch.rand(3, 4), torch.rand(3, 4)# scripted_gate = torch.jit.trace(MyDecisionGate(), x)scripted_gate = torch.jit.script(MyDecisionGate())class MyRNNLoop(torch.nn.Module):    def __init__(self):        super(MyRNNLoop, self).__init__()        self.cell = torch.jit.trace(MyCell(scripted_gate), (x, h))    def forward(self, xs):        h, y = torch.zeros(3, 4), torch.zeros(3, 4)        for i in range(xs.size(0)):            y, h = self.cell(xs[i], h)        return y, hrnn_loop = torch.jit.script(MyRNNLoop())print(rnn_loop.code)print(rnn_loop.cell.dg.code)class WrapRNN(torch.nn.Module):    def __init__(self):        super(WrapRNN, self).__init__()        self.loop = torch.jit.script(MyRNNLoop())    def forward(self, xs):        y, h = self.loop(xs)        return torch.relu(y)traced = torch.jit.trace(WrapRNN(), (torch.rand(10, 3, 4)))print(traced.code)print(traced.loop.cell.dg.code)traced.save('wrapped_rnn.pt')loaded = torch.jit.load('wrapped_rnn.pt')
```

**C++ Fronted API**

* 一种方案是使用C++ API进行完整的训练与模型保存\(torchscript中定义的格式\)
* 另一种方案是使用Python训练并保存\(必须使用torchscript定义的格式保存\), 然后使用C++ API进行导入

## Pytorch Internal

### out=xxx

参考 [stackoverflow](https://discuss.pytorch.org/t/whats-the-benefit-of-using-out-parameter-in-tensor-operations/31728) 问答

### inplace=True

pytorch 中某些函数允许使用 inplace=True，但前提条件是这个 tensor 在反向求导时是不需要的：

例如，对于 relu 函数，`y=relu(x)`，`y` 对于 `x` 的局部导数可以直接通过 `dy_dx=(y>0)` 得到，而无需知道 `x` 的值。因此可以使用：

```python
x = torch.nn.functional.relu(x, inplace=True)
```

另一个例子

准确理解这个问题跟计算图相关，尤其是 backward 与 forward 之间的关系。参考[关于自定义算子的官方文档](https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html)。

```python
import torch

class Square(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Because we are saving one of the inputs use `save_for_backward`
        # Save non-tensors and non-inputs/non-outputs directly on ctx
        ctx.save_for_backward(x)
        return x**2

    @staticmethod
    def backward(ctx, grad_out):
        # A function support double backward automatically if autograd
        # is able to record the computations performed in backward
        x, = ctx.saved_tensors
        return grad_out * 2 * x

# Use double precision because finite differencing method magnifies errors
x = torch.rand(3, 3, requires_grad=True, dtype=torch.double)
torch.autograd.gradcheck(Square.apply, x)
# Use gradcheck to verify second-order derivatives
torch.autograd.gradgradcheck(Square.apply, x)
```

### detach

注意，使用 detach 后，返回的新张量与原来的张量共享内存。详情参考[官方文档](https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html)，摘录如下

> Returned Tensor shares the same storage with the original one. In-place modifications on either of them will be seen, and may trigger errors in correctness checks. IMPORTANT NOTE: Previously, in-place size / stride / storage changes (such as resize_ / resize_as_ / set_ / transpose_) to the returned tensor also update the original tensor. Now, these in-place changes will not update the original tensor anymore, and will instead trigger an error. For sparse tensors: In-place indices / values changes (such as zero_ / copy_ / add_) to the returned tensor will not update the original tensor anymore, and will instead trigger an error.

### non-blocking 参数

参考[问答](https://jovian.ai/forum/t/purpose-of-non-blocking-true-in-tensor-to/14760)、[问答](https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/3)、[pytorch官方例程](https://github.com/pytorch/examples/blob/master/imagenet/main.py#L280-L291)

## 常用函数

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

```
x = torch.tensor([1, 2, 3])
y = x[[0, 2], None]  # 相当于y=x[[0, 2]].view(-1, 1)
```

`torch.nn.functional.normalize`

```python
torch.nn.functional.normalize(input, p=2.0, dim=1, eps=1e-12,out=None)
```

默认对第 1 维进行归一化，例如：

```python
normalize(torch.tensor([[6, 8]])) # [[0.6, 0.8]]
```

## 易错记录

### clone、detach

```python
target = source.clone().detach()
target = source.clone().detach().to("cuda:1").requires_grad_(True)
```

### torch.std 与 torch.var

这两个函数均计算的是无偏估计，以一维数据为例，即 `x` 的形状为 `(d,)`，`torch.std(x)` 的计算公式为：
$$
x.std(x)=torch.std(x)=\sqrt\frac{\sum_{i=1}^{d}(x_i-\bar{x}_i)^2}{d-1}
$$
`torch.var(x)` 的计算方式与 `torch.std(x)` 是一致的，即：
$$
x.var(x)=torch.var(x)=\frac{\sum_{i=1}^{d}(x_i-\bar{x}_i)^2}{d-1}
$$

### torch.nn.LayerNorm（附手动实现）

以二维数据为例，即 `x` 的形状为 `(B, C)`，调用方法为：

```python
# 还原计算过程
import torch
B, C = 3, 4
x = torch.rand([B, C])
weight = torch.rand([C])
bias = torch.rand([C])
eps = 1e-5

out1 = torch.nn.functional.layer_norm(x, (C,), weight, bias, eps=eps)

# 手动计算
mean = x.mean(axis=1)
var = x.var(axis=1)
# 注意此处用的是有偏估计
out2 = (x - mean.view(-1, 1)) / torch.sqrt(var.view(-1, 1)*(C-1)/C+1e-5)
out2 = out2 * weight + bias
```

更复杂的情形可以按如下方式手动实现 `layer_norm`

```python
import torch
B, L, C1, C2 = 1, 2, 4, 3
normalized_shape=[C1, C2]
x = torch.rand([B, L, C1, C2])
weight = torch.rand(normalized_shape)
bias = torch.rand(normalized_shape)
eps = 1e-5


out1 = torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps=eps)

dims = len(normalized_shape)
features = 1
for num in normalized_shape:
    features *= num
mean = x.mean(dim=[-i for i in range(1, dims+1)], keepdim=True)
var = x.var(dim=[-i for i in range(1, dims+1)], keepdim=True)
out2 = (x - mean) / torch.sqrt(var*(features-1)/features+1e-5)
out2 = out2 * weight + bias

(out1-out2).abs().sum()
```

## `torch.nn.Module` 源码剖析

- 版本：torch 1.6.0
- 相关代码：`torch/nn/modules/module.py`

### register 与 hook 机制

可以使用如下方式使得每次调用 `nn.Module` 的 forward 函数时，都将输入打印出来。

```python
import torch
from torch.nn.modules.module import register_module_forward_pre_hook
def custom_pre_forward_hook(module, input):
    print(input)

register_module_forward_pre_hook(custom_pre_forward_hook)

class A(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.count = 0
    def forward(self, x):
        self.count += 1
        return x
x = A()(torch.rand([2]))
```

这种影响全局的注册 hook 的方法有如下几个：

- register_module_forward_pre_hook
- register_module_forward_hook
- register_module_backward_hook

而在 `nn.Module` 的代码中，存在如下与 hook 或 register 有关的函数

- register_buffer
- register_parameter
- register_backward_hook
- register_forward_pre_hook
- register_forward_hook
- _register_state_dict_hook
- _register_load_state_dict_pre_hook

### `__init__` 函数

自定义模型都要继承自 `nn.Module`，并且在子类中一般都有如下操作

```python
class CustomModule(nn.Module):
	def __init__(self):
        super().__init__()
```

相关源码如下：

```python
def __init__(self):
    """
    Initializes internal Module state, shared by both nn.Module and ScriptModule.
    """
    torch._C._log_api_usage_once("python.nn_module")
	
    # 与相关的实例方法的对应关系为
    self.training = True  # train, eval
    self._parameters = OrderedDict()  # register_parameter
    self._buffers = OrderedDict()  # register_buffer
    self._non_persistent_buffers_set = set()
    self._backward_hooks = OrderedDict()  # register_backward_hook
    self._forward_hooks = OrderedDict()  # register_forward_hook
    self._forward_pre_hooks = OrderedDict()  # register_forward_pre_hook
    self._state_dict_hooks = OrderedDict()  # _register_state_dict_hook
    self._load_state_dict_pre_hooks = OrderedDict()  # _register_load_state_dict_pre_hook
    self._modules = OrderedDict()
```

### `__call__`

源码如下：

```python
__call__ : Callable[..., Any] = _call_impl

def _call_impl(self, *input, **kwargs):
    # 先调用全局的hook，后调用特有的hook
    for hook in itertools.chain(
            _global_forward_pre_hooks.values(),  # _global_forward_pre_hooks是一个OrderedDict
            self._forward_pre_hooks.values()):
        result = hook(self, input)
        if result is not None:
            if not isinstance(result, tuple):
                result = (result,)
            input = result
    if torch._C._get_tracing_state():
        result = self._slow_forward(*input, **kwargs)
    else:
        result = self.forward(*input, **kwargs)
    for hook in itertools.chain(
            _global_forward_hooks.values(),
            self._forward_hooks.values()):
        hook_result = hook(self, input, result)
        if hook_result is not None:
            result = hook_result
    if (len(self._backward_hooks) > 0) or (len(_global_backward_hooks) > 0):
        var = result
        while not isinstance(var, torch.Tensor):
            if isinstance(var, dict):
                var = next((v for v in var.values() if isinstance(v, torch.Tensor)))
            else:
                var = var[0]
        grad_fn = var.grad_fn
        if grad_fn is not None:
            for hook in itertools.chain(
                    _global_backward_hooks.values(),
                    self._backward_hooks.values()):
                wrapper = functools.partial(hook, self)
                functools.update_wrapper(wrapper, hook)
                grad_fn.register_hook(wrapper)
    return result
```

## Docker

devel 与 runtime 的主要区别在于后者没有 nvcc

## AllReduce

reduce的含义是规约，即多个数据规约为一个数据，例如求和，求平均，求最大值等。而allreduce指的是将多个机器上的数据跨机器做规约，并且最终的规约值要放在每个机器上各一份。典型的例子是对多个机器求向量和，即：

```
机器1: [1, 2, 3, 4]
机器2: [5, 6, 7, 8]
机器3: [9, 10, 11, 12]
机器4: [13, 14, 15, 16]
```

目标：

```
机器1: [1+5+9+13, 2+6+10+14, 3+7+11+15, 4+8+12+16]
机器2: [1+5+9+13, 2+6+10+14, 3+7+11+15, 4+8+12+16]
机器3: [1+5+9+13, 2+6+10+14, 3+7+11+15, 4+8+12+16]
机器4: [1+5+9+13, 2+6+10+14, 3+7+11+15, 4+8+12+16]
```

### Naive

步骤如下：

- 机器1首先接受来自其他机器的所有数字并计算最终结果

- 将计算结果从机器1分发至其余机器

分析：

- 通信成本

  ```
  机器1：接收12个数据，发送12个数据
  机器2：接收4个数据，发送4个数据
  机器3：接收4个数据，发送4个数据
  机器4：接收4个数据，发送4个数据
  ```

- 计算成本

  ```
  机器1：12次加法
  机器2：0次运算
  机器3：0次运算
  机器4：0次运算
  ```

- 时间成本

  24个数据通信时间+12次加法计算时间

### Ring AllReduce

步骤如下

- Scatter-reduce

  ```
  机器1：[1, 2, 3, 4] (-> 1) (<- 16) (add)  => [1, 2, 3, 20]
  机器2：[5, 6, 7, 8] (-> 6) (<- 1) (add)  => [6, 6, 7, 8]
  机器3：[9, 10, 11, 12] (-> 11) (<- 6) (add)  => [9, 16, 11, 12]
  机器4：[13, 14, 15, 16] (-> 16) (<- 11) (add)  => [13, 14, 26, 16]
  ```

  ```
  机器1：[1, 2, 3, 20] (-> 20) (<- 26) (add)  => [1, 2, 29, 20]
  机器2：[6, 6, 7, 8] (-> 6) (<- 20) (add)  => [6, 6, 7, 28]
  机器3：[9, 16, 11, 12] (-> 16) (<- 6) (add)  => [15, 16, 11, 12]
  机器4：[13, 14, 26, 16] (-> 26) (<- 16) (add)  => [13, 30, 26, 16]
  ```

  ```
  机器1：[1, 2, 29, 20] (-> 29) (<- 30) (add)  => [1, 32, 29, 20]    # 32
  机器2：[6, 6, 7, 28] (-> 28) (<- 29) (add)  => [6, 6, 36, 28]      # 36
  机器3：[15, 16, 11, 12] (-> 15) (<- 28) (add)  => [15, 16, 11, 40] # 40
  机器4：[13, 30, 26, 16] (-> 30) (<- 15) (add)  => [28, 30, 26, 16] # 28
  ```

- Allgather

  类似于第一步，但接收到数据后不做运算，只进行覆盖更新

分析

- 通信成本

  ```
  机器1：接收6个数据，发送6个数据
  机器2：接收6个数据，发送6个数据
  机器3：接收6个数据，发送6个数据
  机器4：接收6个数据，发送6个数据
  ```

- 计算成本

  ```
  机器1：3次加法, 3次覆盖
  机器2：3次加法, 3次覆盖
  机器3：3次加法, 3次覆盖
  机器4：3次加法, 3次覆盖
  ```

- 时间成本

  假定所有机器处理速度完全相同时，花费时间为：12个数据通信时间，3次加法时间，3次覆盖时间

## Tricks and Discusions

### 为什么 torchvision 使用 PIL 而非 CV2？

可参考[stackoverflow问答](https://stackoverflow.com/questions/61346009/why-is-pil-used-so-often-with-pytorch)

目前 torchvision 也可以选择图像处理的后端，按官方说法：accimage 一般比 PIL 要快一些，但支持的操作不完善

```python
torchvision.set_image_backend(backend)
# backend可以选择"PIL"或者"accimage"
```

### pytorch 与 cuda/cudnn 的对应

[pytorch 论坛](https://discuss.pytorch.org/t/how-to-check-if-torch-uses-cudnn/21933/4) 的解答中有如下解释

> Yes, you just need to install the NVIDIA drivers and the binaries will come with the other libs.
> If you want to build from source, you would need to install CUDA, cuDNN etc.

如果采用二进制的形式安装（pip install 属于这一类），那么只需要事先安装好显卡驱动即可，安装包里已经内置了 CUDA 与 cuDNN。这也可能解释了为什么 pytorch 的官方 Docker 镜像例如 `pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime` 标签名写的是 cudnn 7 而实际上包含的 cudnn_version.h 里显示的是 8.2.1 版本。

