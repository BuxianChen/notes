# MMCV/MMDETECT 架构分析

mmcv 是 mmdet 的主要依赖包。mmcv 的特点是广泛地使用配置文件来规范化模型的构建、训练等。实现这一特性的核心是 mmcv 的 `Config` 类与 `Registry` 类。分别定义于 `mmcv/utils/config.py` 与 `mmcv/utils/registry.py` 文件中。

## 典型用法（此处补充一个 mmdet 的例子）：

```python
dataset = DATASET.build(Config.fromfile("xxx.py"))
model = MODEL.build(Config.fromfile("xxx.py"))
runner = RUNNER.build(Config.fromfile("xxx.py"))
```

## mmcv.utils.config.Config 类

基本上是读取配置文件转为字典

## mmcv.utils.registry.Registry 类

目的是希望可以使用类似于如下的方式进行规范化：

```python
# from mmcv.cnn import MODELS
MODELS = Register("model") # 实际上也就是上一行
@MODELS.register_module()
class MyModel:
	pass
MODEL.build(Config.fromfile("xxx.py"))
```

Register 的作用如下，假定 MODELS 为一个 Register 对象，通过对其他的类使用 MODELS.register_module 装饰，在 MODELS 内部维护一个映射表，例如：

```
{"resnet": ResNet, "vgg": VGG}
```

而被装饰的类本身没有被进行任何的修改。

mmcv/utils/registry.py

```python
@staticmethod
def infer_scope():
    # 感觉设计略有不妥，这个函数实际上只能在__init__中被调用
	pass
```

简化版：

```python
class Register:
    def __init__(self, name):
        self.name = name
        self.module_dict = dict()
        
    def register_module(self, module_class=None, name=None):
        if module_class is not None:
            key = module_class.__name__ if name is None else name
            self.module_dict[key] = module_class
            return module_class
        def wrapper(cls):
            key = cls.__name__ if name is None else name
            self.module_dict[key] = cls
            return cls
        return wrapper
        
    def build(self, cfg):
        args = cfg.copy()
        name = args.pop("type")
        assert name in self.module_dict
        return self.module_dict[name](**args)

REG = Register("reg")

@REG.register_module(name="a")
class A:
    def __init__(self, a, b):
        self.a = a
        self.b = b

class B:
    def __init__(self, c):
        self.c = c
B = REG.register_module(B)

print(REG.module_dict)
print(REG.build({"type": "a", "a": 1, "b": 2}))
print(REG.build({"type": "B", "c": 3}))
```

## mmcv.runner.builder.Runner 类



