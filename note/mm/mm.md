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

```python
# mmcv/cnn/builder.py
MODELS = Registry('model', build_func=build_model_from_cfg)
# build_model_from_cfg(cfg)的作用等同于使用cfg的type参数构建实例

# mmdet/models/builder.py
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)

BACKBONES = MODELS
NECKS = MODELS
ROI_EXTRACTORS = MODELS
SHARED_HEADS = MODELS
HEADS = MODELS
LOSSES = MODELS
DETECTORS = MODELS
```

### mmdet/../tools/train.py 脚本源码解析

此脚本为训练脚本，通常利用这个脚本训练模型

```python
# mmdet/../tools/train.py
def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    # 省略若干代码...
    
    # mmdet/models/builder.py
    # def build_detector(cfg, train_cfg=None, test_cfg=None):
    #     # 如果cfg.train_cfg与train_cfg同时被指定会报错, 目的是为了兼容性, 见下面的解释
    #	  # 省略若干代码...
    #     return DETECTORS.build(cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
    # DETECTORS.build实际上就是调用build_model_from_cfg
    # build_model_from_cfg函数的default_args
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)
```

兼容性：在 mmdet 之前的版本里，配置文件中通常按这种方式组织

```python
model = dict(
    # ...
    train_cfg = ...
    test_cfg = ...
)
```

但在此版本的 mmdet 现在的版本里，推荐用这种方式组织

```
model = dict(
    # ...
)
train_cfg = ...
test_cfg = ...
```

#### 运行实例：Yolov3

下面具体介绍一个模型，启动方式为：

```python
# 在与mmdet的同级目录下启动
python tools/train.py configs/yolo/yolov3_d53_mstrain-608_273e_coco.py
```

而简化版的配置文件为（将配置文件的继承关系展平了）：

```python
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]


# model settings
model = dict(
    type='YOLOV3',
    backbone=dict(type='Darknet', ...),
    neck=dict(type='YOLOV3Neck',...),
    bbox_head=dict(type='YOLOV3Head', ...),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='GridAssigner',
            pos_iou_thr=0.5,...)),
    test_cfg=dict(
        score_thr=0.05,
        conf_thr=0.005,
        ...))
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    ...,
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(608, 608),
        ...
        transforms=[
            dict(type='Resize', keep_ratio=True),
            ...,
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))  # 测试时会按照这个路径来进行测试
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(policy='step',warmup='linear',warmup_iters=2000,warmup_ratio=0.1,step=[218, 246])
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=273)
evaluation = dict(interval=1, metric=['bbox'])
```

因此，运行时，首先会执行到以下代码构建模型

```
build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
```

而这行代码最终会执行的类似于：

```python
cfg.model.pop("type")
YOLOV3(**cfg.model) # cfg.model中的键为backbone, neck, bbox_head, train_cfg,test_cfg
# 因此根据下面YOLOV3的定义，配置文件中model里实际上可以增加pretrain与init_cfg
# 备注: pretrain将弃用, 因此最好只添加init_cfg
```

而 `YOLOV3` 类是按如下方式定义的：

```python
# mmdet/models/detectors/yolo.py: 全部代码
@DETECTORS.register_module()
class YOLOV3(SingleStageDetector):
    def __init__(self,backbone,neck,bbox_head,train_cfg=None,
                 test_cfg=None,pretrained=None,init_cfg=None):
        super(YOLOV3, self).__init__(backbone, neck, bbox_head, train_cfg,
                                     test_cfg, pretrained, init_cfg)
# mmdet/models/detectors/single_stage.py:
@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    # 省略具体代码
    pass
# mmdet/models/detectors/base.py:
class BaseDetector(BaseModule, metaclass=ABCMeta):
    # 省略具体代码
    pass
# mmcv/runner/base_module.py
class BaseModule(nn.Module, metaclass=ABCMeta):
    # 省略具体代码
    pass
```

所以实际上只需理解 `SingleStageDetector`，`BaseDetector`，`BaseModule` 这三个类的逻辑即可。

首先，`SingleStageDetector.__init__` 函数会依次调用

```python
super(SingleStageDetector, self).__init__(init_cfg)  #
self.backbone = build_backbone(backbone)  # 等同于 BACKBONES.build(cfg)
self.neck = build_neck(neck)              # 等同于 NECKS.build(cfg)
self.bbox_head = build_head(bbox_head)    # 等同于 HEADS.build(cfg)
```

前面已经提到，伪代码如下，因此最终都回到了 `build_model_from_cfg` 函数的调用：

```
mmcv.MODELS=Registry('model', build_func=build_model_from_cfg)
mmdet.MODELS = Registry('models', parent=mmcv.MODELS)
BACKBONES=NECKS=HEADS=mmdet.MODELS
```



## mmcv.runner.builder.Runner 类





```
python tools/test.py configs/yolo/yolov3_d53_320_273e_coco.py checkpoints/yolov3_d53_320_273e_coco-421362b6.pth --show-dir temp
```



mmdet.



## 一些相对独立的底层代码

### mmcv/runner/base_module.py：BaseModule

mmcv 中类似于 torch.nn.Module 的东西，完整源代码（注释有所修改）如下，可以发现本质上只是给 `nn.Module` 增加了个 `init_weights` 方法。

```python
class BaseModule(nn.Module, metaclass=ABCMeta):
    """Base module for all modules in openmmlab."""

    def __init__(self, init_cfg=None):
        """Initialize BaseModule, inherited from `torch.nn.Module`

        Args:
            init_cfg (dict, optional): Initialization config dict.
        """

        # NOTE init_cfg can be defined in different levels, but init_cfg
        # in low levels has a higher priority.

        super(BaseModule, self).__init__()
        # define default value of init_cfg instead of hard code
        # in init_weight() function
        self._is_init = False
        self.init_cfg = init_cfg

        # Backward compatibility in derived classes
        # if pretrained is not None:
        #     warnings.warn('DeprecationWarning: pretrained is a deprecated \
        #         key, please consider using init_cfg')
        #     self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

    @property
    def is_init(self):
        return self._is_init

    def init_weights(self):
        """Initialize the weights."""
        from ..cnn import initialize

        if not self._is_init:
            if self.init_cfg:
                initialize(self, self.init_cfg)
                if isinstance(self.init_cfg, (dict, ConfigDict)):
                    # Avoid the parameters of the pre-training model
                    # being overwritten by the init_weights
                    # of the children.
                    if self.init_cfg['type'] == 'Pretrained':
                        return

            for m in self.children():
                if hasattr(m, 'init_weights'):
                    m.init_weights()
            self._is_init = True
        else:
            warnings.warn(f'init_weights of {self.__class__.__name__} has '
                          f'been called more than once.')

    def __repr__(self):
        s = super().__repr__()
        if self.init_cfg:
            s += f'\ninit_cfg={self.init_cfg}'
        return s
```

以下是一些继承 `mmcv/runner/base_

#### mmdet/models/detectors/base.py：BaseDetector

```
```


