# MMCV/MMDETECT 架构分析

mmcv 是 mmdet 的主要依赖包。mmcv 的特点是广泛地使用配置文件来规范化模型的构建、训练等。实现这一特性的核心是 mmcv 的 `Config` 类与 `Registry` 类。分别定义于 `mmcv/utils/config.py` 与 `mmcv/utils/registry.py` 文件中。以下架构分析**针对的版本为**：

- mmcv-full 1.3.9
- mmdetection （git commit id: 522eb9ebd7df0944b2a659354f01799895df74ce，版本为2.14~2.15之间）

## 典型用法（此处补充一个 mmdet 的例子）：

```python
dataset = DATASET.build(Config.fromfile("xxx.py"))
model = MODEL.build(Config.fromfile("xxx.py"))
runner = RUNNER.build(Config.fromfile("xxx.py"))
```

## mmcv.utils.config.Config 类

基本上是读取配置文件转为字典

```python
# source code: mmcv.utils.config.py
# 截取了一部分Config类的代码
class Config:
    def __init__(self, cfg_dict=None, cfg_text=None, filename=None):
        if cfg_dict is None:
            cfg_dict = dict()
        """cfg_dict的类型必须为字典（或者是其子类）。特别地，下面出现的ConfigDict继承自addict.Dict，
        而addict.Dict继承dict。addict的Dict类主要是允许使用如下方式进行数据访问，而ConfigDict对Dict的修改较小。
        >>> d = Dict({"a": 1})
        >>> print(d.a, d["a"])"""
        elif not isinstance(cfg_dict, dict):
            raise TypeError('cfg_dict must be a dict, but '
                            f'got {type(cfg_dict)}')
        for key in cfg_dict:
            """
            RESERVED_KEYS也定义在本文件中:
            RESERVED_KEYS = ['filename', 'text', 'pretty_text']
            cfg_dict中不允许有这些键, 原因参见__getattr__与filename的定义
            """
            if key in RESERVED_KEYS:
                raise KeyError(f'{key} is reserved for config file')

        super(Config, self).__setattr__('_cfg_dict', ConfigDict(cfg_dict))
        super(Config, self).__setattr__('_filename', filename)
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, 'r') as f:
                text = f.read()
        else:
            text = ''
        super(Config, self).__setattr__('_text', text)
        
    # 通过内部的ConfigDict对象得到属性
	def __getattr__(self, name):
        return getattr(self._cfg_dict, name)
    
    @property
    def filename(self):
        return self._filename
    
    @staticmethod
    def fromfile(filename,
                 use_predefined_variables=True,
                 import_custom_modules=True):
        cfg_dict, cfg_text = Config._file2dict(filename,
                                               use_predefined_variables)
        if import_custom_modules and cfg_dict.get('custom_imports', None):
            import_modules_from_strings(**cfg_dict['custom_imports'])
        return Config(cfg_dict, cfg_text=cfg_text, filename=filename)
```

mmdetection 内置的模型通常使用配置文件，放置在 `mmdetection/configs` 文件夹下，

### 进阶：Config.fromfile 方法的实现细节

<details>
<summary>
成功代码：
</summary>

```python
class Solution:
    def solve(self):
        return None
```
</details>


```python
# source code: mmcv.utils.config.py
import os
import os.path as osp
import sys
import tempfile

class Config:
    @staticmethod
    def fromfile(filename,
                 use_predefined_variables=True,
                 import_custom_modules=True):
        cfg_dict, cfg_text = Config._file2dict(filename,
                                               use_predefined_variables)
        if import_custom_modules and cfg_dict.get('custom_imports', None):
            import_modules_from_strings(**cfg_dict['custom_imports'])
        return Config(cfg_dict, cfg_text=cfg_text, filename=filename)
    
	@staticmethod
	def _file2dict(filename, use_predefined_variables=True):
	    filename = osp.abspath(osp.expanduser(filename))
	    check_file_exist(filename)
	    fileExtname = osp.splitext(filename)[1]
	    if fileExtname not in ['.py', '.json', '.yaml', '.yml']:
	        raise IOError('Only py/yml/yaml/json type are supported now!')
	
	    with tempfile.TemporaryDirectory() as temp_config_dir:
	        temp_config_file = tempfile.NamedTemporaryFile(
	            dir=temp_config_dir, suffix=fileExtname)
	        if platform.system() == 'Windows':
	            temp_config_file.close()
	        temp_config_name = osp.basename(temp_config_file.name)
	        # Substitute predefined variables
	        if use_predefined_variables:
	            Config._substitute_predefined_vars(filename,
	                                               temp_config_file.name)
	        else:
	            shutil.copyfile(filename, temp_config_file.name)
	        # Substitute base variables from placeholders to strings
	        base_var_dict = Config._pre_substitute_base_vars(
	            temp_config_file.name, temp_config_file.name)
	
	        if filename.endswith('.py'):
	            temp_module_name = osp.splitext(temp_config_name)[0]
	            sys.path.insert(0, temp_config_dir)
	            Config._validate_py_syntax(filename)
	            mod = import_module(temp_module_name)
	            sys.path.pop(0)
	            cfg_dict = {
	                name: value
	                for name, value in mod.__dict__.items()
	                if not name.startswith('__')
	            }
	            # delete imported module
	            del sys.modules[temp_module_name]
	        elif filename.endswith(('.yml', '.yaml', '.json')):
	            import mmcv
	            cfg_dict = mmcv.load(temp_config_file.name)
	        # close temp file
	        temp_config_file.close()
	
	    cfg_text = filename + '\n'
	    with open(filename, 'r', encoding='utf-8') as f:
	        # Setting encoding explicitly to resolve coding issue on windows
	        cfg_text += f.read()
	
	    if BASE_KEY in cfg_dict:
	        cfg_dir = osp.dirname(filename)
	        base_filename = cfg_dict.pop(BASE_KEY)
	        base_filename = base_filename if isinstance(
	            base_filename, list) else [base_filename]
	
	        cfg_dict_list = list()
	        cfg_text_list = list()
	        for f in base_filename:
	            _cfg_dict, _cfg_text = Config._file2dict(osp.join(cfg_dir, f))
	            cfg_dict_list.append(_cfg_dict)
	            cfg_text_list.append(_cfg_text)
	
	        base_cfg_dict = dict()
	        for c in cfg_dict_list:
	            if len(base_cfg_dict.keys() & c.keys()) > 0:
	                raise KeyError('Duplicate key is not allowed among bases')
	            base_cfg_dict.update(c)
	
	        # Subtitute base variables from strings to their actual values
	        cfg_dict = Config._substitute_base_vars(cfg_dict, base_var_dict,
	                                                base_cfg_dict)
	
	        base_cfg_dict = Config._merge_a_into_b(cfg_dict, base_cfg_dict)
	        cfg_dict = base_cfg_dict
	
	        # merge cfg_text
	        cfg_text_list.append(cfg_text)
	        cfg_text = '\n'.join(cfg_text_list)
	
	    return cfg_dict, cfg_text
```



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

### mmcv 与 mmdetection 中的 Registry 实例：

```
```



## mmcv.runner.builder.Runner 类





```
python tools/test.py configs/yolo/yolov3_d53_320_273e_coco.py checkpoints/yolov3_d53_320_273e_coco-421362b6.pth --show-dir temp
```



