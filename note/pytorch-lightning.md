# Lightning

## 参考资料

- [pytorch-lightning 101](https://www.youtube.com/playlist?list=PLaMu-SDt_RB5NUm67hU2pdE75j6KaIOv2): 视频课程, 可以用来理解概念, 共 4 小节课程, 其中第 3 小节是 pytorch-lightning 的基本用法, 第 4 小节介绍了 pytorch-lightning 的实现细节

- [LightningLite](https://pytorch-lightning.readthedocs.io/en/latest/starter/lightning_lite.html): 轻量版的 pytorch-lighting, 目前(2022.9.29)并未完全成熟. 用于尽量做很少的代码改动, 快速将 pytorch 训练代码进行转换, 好处是可以很快地将写好的单 GPU 或 CPU 训练流程变得可以自动支持多卡训练, fp16 训练等.

- [pytorch-lightning with huggingface transfomers](https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/text-transformers.html)


## 使用


### 疑惑

- `LightningModule` 中的 `self.log(...)` 是指什么?(猜测用于传给torchmetric, 类似于tf的Metric) 似乎最终调用的是`pytorch_lightning.trainer.connectors.logger_connector.result._ResultCollection.log()`
    - 此函数体内涉及到`lightning_utilities.core.apply_func.apply_to_collections`

### Pytorch vs Lightning

- Dataset, DataLoader: 在 Lightning 可以沿用, 或者使用 `LightningDataModule`, 多卡训练时, Dataloader 所需的 DistributedSampler 在 Lightning 中无需手动写
- nn.Module: 在 Lightning 使用 `LightningModule`, 需要提供 `forward`, `training_step`, `configure_optimizers` 方法
- 训练过程: 
    - for loop: 在 Lightning 无需自己写 for loop
    - loss backward: 在 Lightning 中可以让 `training_step` 返回 loss, 自动进行 backward, 或者也可以手工写 backward 的执行过程
- DDP, AMP: 在 Lightning 中用 `Trainer` 的初始化参数指定
- 模型加载与保存: 最简单的用法是 Lightning 中用 `Trainer` 自动处理, 高级用法是初始化 `Trainer` 时增加 `pytorch_lightning.callbacks.ModelCheckpoint` 这个 callback, 更复杂的用法是关闭 `Trainer` 的模型保存功能(`enable_checkpointing=False`), 在 `LightningModule` 的 `training_epoch_end` 或者 `validation_epoch_end` 中检测当前的 local_rank, 只在 local_rank 为0的进程上做保存模型的工作
- 控制台打印: 可以使用 python 本身的 `logging` 模块进行, 参考[官方文档](https://pytorch-lightning.readthedocs.io/en/stable/common/console_logs.html)

### pytorch_lightning.LightningModule

代码参考自: [pytorch-lightning with huggingface transfomers](https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/text-transformers.html)

```python
class GLUETransformer(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        task_name: str,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        **kwargs,
    ):
        super().__init__()

        # 此函数用于保存__init__函数的入参至self.hparams, 含义是收集所有的超参数,推荐在此处使用
        # 如果__init__函数的入参中有torch.nn.Module, 可以设置参数ignore将其忽略
        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.metric = datasets.load_metric(
            "glue", self.hparams.task_name, experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        # 返回一个标量版的loss即可, 或者返回一个字典, 字典中有一个键值对为{"loss": loss}
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]

        return {"loss": val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        if self.hparams.task_name == "mnli":
            for i, output in enumerate(outputs):
                # matched or mismatched
                split = self.hparams.eval_splits[i].split("_")[-1]
                preds = torch.cat([x["preds"] for x in output]).detach().cpu().numpy()
                labels = torch.cat([x["labels"] for x in output]).detach().cpu().numpy()
                loss = torch.stack([x["loss"] for x in output]).mean()
                self.log(f"val_loss_{split}", loss, prog_bar=True)
                split_metrics = {
                    f"{k}_{split}": v for k, v in self.metric.compute(predictions=preds, references=labels).items()
                }
                self.log_dict(split_metrics, prog_bar=True)
            return loss

        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,  # 注意: 此处可以使用trainer的变量
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
```

#### inference without pytorch-lightning package

为了更好地模块化, 建议采用如下方式组织代码

```python
# 存疑: 官方演示视频中此处继承的是LightningModule
class TorchModule(torch.nn.Module):
    def __init__(self, **model_hparams):
        ...
    def forward(self, input):
        ...
class PLModule(LightningModule)
    def __init__(self, model, **kwargs):
        self.model = model
    def training_step(self, batch, batch_idx):
        ...
    def validation_step(self, batch, batch_idx):
        ...
    def configure_optimizers(self):
        ...
```

#### 控制backward的逻辑

更多详细内容参考[官方文档](https://pytorch-lightning.readthedocs.io/en/stable/common/optimization.html)

如果需要控制backward的逻辑, 需要做以下事情
- 在`__init__`中设置`self.automatic_optimization=False`
- 在`training_step`中使用如下API:
    - 使用 `self.optimizers()` 获取所有优化器
    - 使用 `optimizer.zero_grad()` 清除梯度
    - 使用 `self.manual_backward(loss)` 而不要使用 `loss.backward()`
    - 使用 `optimizer.step()`

#### 多个optimizer与scheduler

建议 `configure_optimizers` 函数按如下方式返回
```python
(
    {
        "optimizer": optimizer_1,
        "lr_scheduler": {
            "scheduler": scheduler_1,
            "interval": "step",
            "frequency": 1,
        }
    },
    {
        "optimizer": optimizer_2,
        "lr_scheduler": {
            "scheduler": scheduler_2,
            "interval": "step",
            "frequency": 1,
        }
    },
)
```

#### current rank

```
def training_step(self, batch, batch_idx):
    self.global_rank
    self.local_rank
```

#### training_step 出入参

入参: 
出参: 返回一个标量版的loss即可, 或者返回一个字典, 字典中有一个键值对为{"loss": loss}

```python
@dataclass
class OutputResult:
    def asdict(self) -> Dict[str, Any]:
        raise NotImplementedError

# src/pytorch_lightning/loops/optimization/optimizer_loop.py
class OptimizerLoop(Loop[_OUTPUTS_TYPE]):
    """Runs over a sequence of optimizers.

    This loop implements what is known in Lightning as Automatic Optimization.
    """

    output_result_cls = ClosureResult
    def _training_step(self, kwargs: OrderedDict) -> ClosureResult:
        ...
        # 注: training_step_output即为training_step的返回结果
        result = self.output_result_cls.from_training_step_output(training_step_output, self.trainer.accumulate_grad_batches)
        ...


@dataclass
class ClosureResult(OutputResult):
    """A container to hold the result of a :class:`Closure` call.

    It is created from the output of :meth:`~pytorch_lightning.core.module.LightningModule.training_step`.

    Attributes:
        closure_loss: The loss with a graph attached.
        loss: A detached copy of the closure loss.
        extra: Any keys other than the loss returned.
    """

    closure_loss: Optional[Tensor]
    loss: Optional[Tensor] = field(init=False, default=None)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._clone_loss()

    def _clone_loss(self) -> None:
        if self.closure_loss is not None:
            # the loss will get scaled for amp. avoid any modifications to it
            self.loss = self.closure_loss.detach().clone()

    @classmethod
    def from_training_step_output(
        cls, training_step_output: Optional[STEP_OUTPUT], normalize: int = 1
    ) -> "ClosureResult":
        closure_loss, extra = None, {}

        if isinstance(training_step_output, dict):
            # this should not modify the `training_step_output`, as the user could be using it after `training_step_end`
            closure_loss = training_step_output.get("loss")
            if closure_loss is None:
                raise MisconfigurationException(
                    "In automatic_optimization, when `training_step` returns a dict, the 'loss' key needs to be present"
                )
            extra = {k: v for k, v in training_step_output.items() if k not in ("loss", "hiddens")}
        elif isinstance(training_step_output, Tensor):
            closure_loss = training_step_output
        elif training_step_output is not None:
            raise MisconfigurationException(
                "In automatic optimization, `training_step` must return a Tensor, "
                "a dict, or None (where the step will be skipped)."
            )

        if closure_loss is not None:
            # accumulate the loss. If ``accumulate_grad_batches == 1``, no effect
            # note: avoid in-place operation `x /= y` here on purpose
            closure_loss = closure_loss / normalize

        return cls(closure_loss, extra=extra)

    def asdict(self) -> Dict[str, Any]:
        return {"loss": self.loss, **self.extra}
```

#### training_step 伪代码(后续源码分析也可参照此)

参考[官方文档](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#hooks), 摘录如下:

```python
def fit(self):
    if global_rank == 0:
        # prepare data is called on GLOBAL_ZERO only
        prepare_data()

    configure_callbacks()

    with parallel(devices):
        # devices can be GPUs, TPUs, ...
        train_on_device(model)


def train_on_device(model):
    # called PER DEVICE
    on_fit_start()
    setup("fit")
    configure_optimizers()

    # the sanity check runs here

    on_train_start()
    for epoch in epochs:
        fit_loop()
    on_train_end()

    on_fit_end()
    teardown("fit")


def fit_loop():
    on_train_epoch_start()

    for batch in train_dataloader():
        on_train_batch_start()

        on_before_batch_transfer()
        transfer_batch_to_device()
        on_after_batch_transfer()

        training_step()

        on_before_zero_grad()
        optimizer_zero_grad()

        on_before_backward()
        backward()
        on_after_backward()

        on_before_optimizer_step()
        configure_gradient_clipping()
        optimizer_step()

        on_train_batch_end()

        if should_check_val:
            val_loop()
    # end training epoch
    training_epoch_end()

    on_train_epoch_end()


def val_loop():
    on_validation_model_eval()  # calls `model.eval()`
    torch.set_grad_enabled(False)

    on_validation_start()
    on_validation_epoch_start()

    val_outs = []
    for batch_idx, batch in enumerate(val_dataloader()):
        on_validation_batch_start(batch, batch_idx)

        batch = on_before_batch_transfer(batch)
        batch = transfer_batch_to_device(batch)
        batch = on_after_batch_transfer(batch)

        out = validation_step(batch, batch_idx)

        on_validation_batch_end(batch, batch_idx)
        val_outs.append(out)

    validation_epoch_end(val_outs)

    on_validation_epoch_end()
    on_validation_end()

    # set up for train
    on_validation_model_train()  # calls `model.train()`
    torch.set_grad_enabled(True)
```

#### `training_step`, `validation_step`, `test_step`, `predict_step`

- training_step: 训练过程, batch中应包含x与y, 被trainer.fit调用
- validation_step: 验证过程, batch中应包含x与y, 通常的每个epoch结束后被trainer.fit调用
- test_step: 训练过程, batch中应包含x与y, 在trainer.fit中不被调用, 被trainer.test调用
- predict_step: 在不定义predict_step的情况下, trainer.pred会调用model.forward, 否则会调用predict_step, 因此batch中只包含x即可

```python
def training_step(batch, batch_idx, optimizer_idx, hiddens):
    # 返回可以是三种:(1) loss tensor (2) dict, 但必须包含"loss"这个key (3) None, 跳过此次training_step的过程, 一般用于手动backward

def validation_step(batch, batch_idx, dataloader_idx):
    # 返回可以是(1)tensor (2)dict of tensor (3) None, 跳过此次validation_step

def test_step(batch, batch_idx, dataloader_id):
    # 返回可以是(1)tensor (2)dict of tensor (3) None, 跳过此次test_step

def predict_step(batch, batch_idx, dataloader_id):
    # 返回是Any
```

#### save checkpoint advanced


**方式1**
默认情况下, 会自动保存模型(只保存最新的), 参考[官方文档](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#enable-checkpointing)

> By default Lightning saves a checkpoint for you in your current working directory, with the state of your last training epoch, Checkpoints capture the exact value of all parameters used by a model. To disable automatic checkpointing, set this to False.

```python
# default used by Trainer, saves the most recent model to a single checkpoint after each epoch
trainer = Trainer(enable_checkpointing=True)

# turn off automatic checkpointing
trainer = Trainer(enable_checkpointing=False)
```

备注: 这种情况下默认会保存在`lightening_logs/version_n/checkpoints` 目录中, 且会保存许多其他非模型权重的东西

```python
import torch
d = torch.load("lightening_logs/version_n/checkpoints/xxx.ckpt")
d.keys()
# epoch, global_step, pytorch_lightening_version, state_dict, loops, callbacks, optimizer_states, lr_schedulers
# 其中state_dict为模型的权重
```

**方式2**
trainer 中传入的callbacks包含一个 `ModelCheckpoint` 实例, 参考[官方文档](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#enable-checkpointing)。用于指定保存路径，保留多少个checkpoint，是否只保留权重，是否根据某个特定的监控值保存最优模型等

```python
from pytorch_lightning.callbacks import ModelCheckpoint

# Init ModelCheckpoint callback, monitoring 'val_loss'
checkpoint_callback = ModelCheckpoint(monitor="val_loss")

# Add your callback to the callbacks list
trainer = Trainer(callbacks=[checkpoint_callback])
```

**方式3**
手写, 在各个hook中加上一些条件进行模型保存

```python
from pytorch_lightning import LightningModule, Trainer
import torch
class MyModel:
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(64, 32)
    def training_epoch_end(self, training_outs):
        if self.current_epoch in [0, 2] and self.local_rank == 0:
            torch.save(self.layer.state_dict(), f"epoch_{self.current_epoch}.pth")
model = MyModel()
trainer = Trainer(max_epochs=4, gpus=2, enable_checkpointing=False)
```

**方式4(不确定): 继承ModelCheckpoint**

### pytorch_lightning.Trainer

```
trainer = Trainer()
trainer.fit(model)
```


### pytorch_lightning.LightningDataModule

```python
import torch
from pytorch_lightning import LightningDataModule

class MyDataset(torch.utils.data.Dataset):
    ...
    def __getitem__(self, idx):
        return transform(data[idx])

class MyDataModule(LightningDataModule):

    def __init__(
        self,
        model_name_or_path: str,
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage: str):
        # 此部分代码在多卡场景下会被每个进程执行, 因此可以设置变量
        # 建议设置self.dataset
        self.datasets = {
            "train": MyDataset(...)
            "val": MyDataset(...)
        }

    def prepare_data(self):
        # 此部分代码仅在rank0上运行, 建议不要设置类的属性
        # 建议做一些数据的转换工作, 例如切分数据至train/val文件夹,将数据tokenizer化保存
        pass

    def train_dataloader(self):
        # 此部分代码在多卡场景下会被每个进程执行
        # 建议dataset在self.setup方法中设定, 此处直接使用torch.utils.data.DataLoader进行包装
        return torch.utils.data.DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset["val"], batch_size=self.train_batch_size, shuffle=False)
```




## 源码阅读

OpenMMLab对pytorch-lightning也有一篇源码解读文章: https://zhuanlan.zhihu.com/p/389271556

目标: 理解如下代码片段的执行过程

```python
from pytorch_lightning import LightModule, Trainer
model = MyModule(...)  # MyModule 继承自 LightModule
trainer = Trainer(...)  # max_steps, min_steps 等参数
trainer.fit(model, train_dataloaders=None, val_dataloaders=None, datamodule=None,ckpt_path=None)
```

### `LightningModule`

源码中关于 `LightningModule` 类的定义继承自了多个父类, 特别注意它也继承自`torch.nn.Module`。因此需要先对几个父类的代码做个了解

```python
class LightningModule(
    _DeviceDtypeModuleMixin,
    HyperparametersMixin,
    ModelIO,
    ModelHooks,
    DataHooks,
    CheckpointHooks,
    Module,
):
    ...
```

**HyperparametersMixin**

主要使用的方式如下参考[官网例子](https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/text-transformers.html): 通过调用 `self.save_hyperparameters` 方法, 将所有 `__init__` 的传参保存到`self._hparams`中

```python
class MyModule(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        learning_rate: float = 2e-5,
        train_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        # MyModule中其他地方可以使用self.hparams.num_labels
        self.save_hyperparameters(ignore=["model_name_or_path"])
model = MyModule("x.pth", 10)
print(model.hparams)  # pytorch_lightning.utilities.parsing.AttributeDict
# {"num_labels": 10, "learning_rate": 2e-5, |train_batch_size": 32}
```

未完待续...


### `Trainer.__init__`

`Trainer` 类没有父类, 直接继承自 `object`.

第 2 行代码: `trainer = Trainer(...)` 的源码如下:

```python
# src/pytorch_lightning/trainer/trainer.py:Trainer
@_defaults_from_env_vars
    def __init__(self, logger=True, ...)  # 共有约50个参数
```

首先解释一下这个装饰器的作用:

利用 `os.environ.get` 方法获取形如 `PL_TRAINER_{XXX}` 环境变量, 并用环境变量的值取代 被装饰的函数(上面的例子中为`Trainer.__init__`函数)中的默认值. 即最终第 2 行代码参数设定的优先顺序为:

- 实参 > 环境变量 > 函数定义中形参的默认值



<details>
<summary>
<hidden_block>
装饰器`_defaults_from_env_vars`的具体实现
</hidden_block>
</summary>

```python
# src/pytorch_lightning/utilities/argparse.py
def _defaults_from_env_vars(fn: _T) -> _T:
    @wraps(fn)  # 注: functools.wraps
    def insert_env_defaults(self: Any, *args: Any, **kwargs: Any) -> Any:
        cls = self.__class__  # get the class
        if args:  # in case any args passed move them to kwargs
            # parse only the argument names
            cls_arg_names = [arg[0] for arg in get_init_arguments_and_types(cls)]
            # convert args to kwargs
            kwargs.update(dict(zip(cls_arg_names, args)))  # 注: 此处的kwargs为实参
        env_variables = vars(parse_env_variables(cls))  # 注: 此处为从环境变量处解析得到的默认值
        # update the kwargs by env variables
        # 注: 这里第2项中的键值对会覆盖第1项的键值对, 因此优先级为实参>环境变量>函数定义中的默认值
        kwargs = dict(list(env_variables.items()) + list(kwargs.items()))

        # all args were already moved to kwargs
        return fn(self, **kwargs)

    return cast(_T, insert_env_defaults)  # 注: typing.cast
```

理解上面的代码所需要的 Python 知识如下:

- `functools.wraps`装饰器的作用
- `vars` 内置函数的作用
- `typing.cast` 函数的作用: 参考[stackoverflow](https://stackoverflow.com/questions/51457563/what-does-typing-cast-do-in-python),此函数只在类型检查时起作用, 而在实际运行时什么都不做, 直接将入参返回


上述代码中进一步调用了如下两段代码

```python
# src/pytorch_lightning/utilities/argparse.py
def parse_env_variables(cls: _ARGPARSE_CLS, template: str = "PL_%(cls_name)s_%(cls_argument)s") -> Namespace:
    """Parse environment arguments if they are defined.

    Examples:

        >>> from pytorch_lightning import Trainer
        >>> parse_env_variables(Trainer)
        Namespace()
        >>> import os
        >>> os.environ["PL_TRAINER_GPUS"] = '42'
        >>> os.environ["PL_TRAINER_BLABLABLA"] = '1.23'
        >>> parse_env_variables(Trainer)
        Namespace(gpus=42)
        >>> del os.environ["PL_TRAINER_GPUS"]
    """
    cls_arg_defaults = get_init_arguments_and_types(cls)

    env_args = {}
    for arg_name, _, _ in cls_arg_defaults:
        env = template % {"cls_name": cls.__name__.upper(), "cls_argument": arg_name.upper()}
        val = os.environ.get(env)
        if not (val is None or val == ""):
            # todo: specify the possible exception
            with suppress(Exception):  # 注: contextlib.suppress
                # converting to native types like int/float/bool
                val = literal_eval(val)  # 注: ast.literal_eval
            env_args[arg_name] = val
    return Namespace(**env_args)


def get_init_arguments_and_types(cls: _ARGPARSE_CLS) -> List[Tuple[str, Tuple, Any]]:
    r"""Scans the class signature and returns argument names, types and default values.

    Returns:
        List with tuples of 3 values:
        (argument name, set with argument types, argument default value).

    Examples:

        >>> from pytorch_lightning import Trainer
        >>> args = get_init_arguments_and_types(Trainer)

    """
    cls_default_params = inspect.signature(cls).parameters  # 注
    name_type_default = []
    for arg in cls_default_params:
        arg_type = cls_default_params[arg].annotation
        arg_default = cls_default_params[arg].default
        try:
            arg_types = tuple(arg_type.__args__)
        except (AttributeError, TypeError):
            arg_types = (arg_type,)

        name_type_default.append((arg, arg_types, arg_default))

    return name_type_default
```

理解上面的代码所需要的 Python 知识如下:
- 内置模块 `inspect` 相关知识: 此处仅用到 `inspect.signature(callable).parameters`, 用于获取函数或类的`__init__`方法定义中的变量名, 变量类型, 以及默认值
- `contextlib.suppress`: 参考[stackoverflow](https://stackoverflow.com/questions/34566806/why-use-contextlib-suppress-as-opposed-to-try-except-with-pass), 这两种写法基本等价:
    ```python
    # 写法一:
    with contextlib.suppress(ValueError):
        x = int('a')
    # 写法二:
    try:
        x = int('a')
    except ValueError:
        pass
    ```
- 内置模块 `ast.literal_eval`: 此函数接受的参数为一个合法的字符串形式的python数据, 例如:

    ```python
    ast.literal_eval("['a', 'b']")  # 返回列表: ["a", "b"]
    ast.literal_eval("'a'")  # 返回字符串: "a"
    ast.literal_eval("1")  # 返回整数: 1
    ast.literal_eval("1.2")  # 返回浮点数: 1.2
    ast.literal_eval("1+1")  # 报错
    ast.literal_eval("a")  # 报错
    ```

    即功能弱于内置函数`eval`, 官方文档中建议一切能用 `ast.literal_eval` 代替 `eval` 的地方, 都使用 `ast.literal_eval`, 无法替代的情况下, 应该选择其他实现方式, 而不能依赖 `eval`

</details>


接下来进入 `Trainer.__init__` 函数的函数体, 如下:

```python
# 
def __init__(self, logger, ....):  # 注: 一共有约50个参数
    super().__init__()
    Trainer._log_api_event("init")
    log.detail(f"{self.__class__.__name__}: Initializing trainer with parameters: {locals()}")
    self.state = TrainerState()

    # init connectors
    self._data_connector = DataConnector(self, multiple_trainloader_mode)

    self._accelerator_connector = AcceleratorConnector(
        num_processes=num_processes,
        devices=devices,
        tpu_cores=tpu_cores,
        ipus=ipus,
        accelerator=accelerator,
        strategy=strategy,
        gpus=gpus,
        num_nodes=num_nodes,
        sync_batchnorm=sync_batchnorm,
        benchmark=benchmark,
        replace_sampler_ddp=replace_sampler_ddp,
        deterministic=deterministic,
        auto_select_gpus=auto_select_gpus,
        precision=precision,
        amp_type=amp_backend,
        amp_level=amp_level,
        plugins=plugins,
    )
    self._logger_connector = LoggerConnector(self)
    self._callback_connector = CallbackConnector(self)
    self._checkpoint_connector = CheckpointConnector(self, resume_from_checkpoint)
    self._signal_connector = SignalConnector(self)
    self.tuner = Tuner(self)

    fit_loop = FitLoop(min_epochs=min_epochs, max_epochs=max_epochs)
    training_epoch_loop = TrainingEpochLoop(min_steps=min_steps, max_steps=max_steps)
    # 注: 执行的函数体为: fit_loop.epoch_loop=training_epoch_loop
    fit_loop.connect(epoch_loop=training_epoch_loop)

    # default .fit() loop
    self.fit_loop = fit_loop

    # default .validate() loop
    self.validate_loop = EvaluationLoop()

    # default .test() loop
    self.test_loop = EvaluationLoop()

    # default .predict() loop
    self.predict_loop = PredictionLoop()

    # set when a checkpoint is loaded via `Trainer.{fit,validate,test,predict}`.
    self._ckpt_path: Optional[str] = None

    # init callbacks
    # Declare attributes to be set in _callback_connector on_trainer_init
    self._callback_connector.on_trainer_init(
        callbacks,
        enable_checkpointing,
        enable_progress_bar,
        default_root_dir,
        enable_model_summary,
        max_time,
        accumulate_grad_batches,
    )

    # hook
    self._call_callback_hooks("on_init_start")

    # init data flags
    self.check_val_every_n_epoch: int
    self._data_connector.on_trainer_init(
        val_check_interval,
        reload_dataloaders_every_n_epochs,
        check_val_every_n_epoch,
    )

    # gradient clipping
    if gradient_clip_val is not None and not isinstance(gradient_clip_val, (int, float)):
        raise TypeError(f"`gradient_clip_val` should be an int or a float. Got {gradient_clip_val}.")

    if gradient_clip_algorithm is not None and not GradClipAlgorithmType.supported_type(
        gradient_clip_algorithm.lower()
    ):
        raise MisconfigurationException(
            f"`gradient_clip_algorithm` {gradient_clip_algorithm} is invalid. "
            f"Allowed algorithms: {GradClipAlgorithmType.supported_types()}."
        )

    # gradient norm tracking
    if track_grad_norm != -1 and not (
        (isinstance(track_grad_norm, (int, float)) or track_grad_norm == "inf") and float(track_grad_norm) > 0
    ):
        raise MisconfigurationException(
            f"`track_grad_norm` must be a positive number or 'inf' (infinity norm). Got {track_grad_norm}."
        )

    self.gradient_clip_val: Union[int, float] = gradient_clip_val
    self.gradient_clip_algorithm: Optional[GradClipAlgorithmType] = (
        GradClipAlgorithmType(gradient_clip_algorithm.lower()) if gradient_clip_algorithm is not None else None
    )
    self.track_grad_norm: float = float(track_grad_norm)

    self._detect_anomaly: bool = detect_anomaly
    self._setup_on_init()

    # configure tuner
    self.tuner.on_trainer_init(auto_lr_find, auto_scale_batch_size)

    # configure profiler
    setup._init_profiler(self, profiler)

    # init logger flags
    self._loggers: List[Logger]
    self._logger_connector.on_trainer_init(logger, log_every_n_steps, move_metrics_to_cpu)

    # init debugging flags
    self.val_check_interval: Union[int, float]
    self.num_sanity_val_steps: Union[float, int]
    setup._init_debugging_flags(
        self,
        limit_train_batches,
        limit_val_batches,
        limit_test_batches,
        limit_predict_batches,
        fast_dev_run,
        overfit_batches,
        val_check_interval,
        num_sanity_val_steps,
    )

    # Callback system
    self._call_callback_hooks("on_init_end")
```

此处代码相对复杂, 先简要分析几处:


**_callback_connector: CallbackConnector**

推测: 这种`XXXConnector`类的作用基本上就是给`Trainer`添加一些属性, 不知道为啥不直接写在`Trainer`的内部(也许是写在Trainer内部, Trainer类的定义会变得很冗长?看源码长度`pytorch_ligtening/trainer/trainer.py`本身已有2000多行, 如果这些Connector也写在Trainer里,估计会更长)

```python
class CallbackConnector:
    ...
    def on_trainer_init(
        self,
        callbacks: Optional[Union[List[Callback], Callback]],
        enable_checkpointing: bool,
        enable_progress_bar: bool,
        default_root_dir: Optional[str],
        enable_model_summary: bool,
        max_time: Optional[Union[str, timedelta, Dict[str, int]]] = None,
        accumulate_grad_batches: Optional[Union[int, Dict[int, int]]] = None,
    ) -> None:
        # init folder paths for checkpoint + weights save callbacks
        self.trainer._default_root_dir = default_root_dir or os.getcwd()

        # init callbacks
        if isinstance(callbacks, Callback):
            callbacks = [callbacks]
        self.trainer.callbacks = callbacks or []

        # configure checkpoint callback
        # pass through the required args to figure out defaults
        # 注: self.trainer.callbacks增加pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint
        self._configure_checkpoint_callbacks(enable_checkpointing)

        # configure the timer callback.
        # responsible to stop the training when max_time is reached.
        # 注: max_time为程序最大运行时间,如果设置,则增加pytorch_lightning.callbacks.timer.Timer
        self._configure_timer_callback(max_time)

        # init progress bar
        # 注: 增加pytorch_lightning.callbacks.progress.tqdm_progress.TQDMProgressBar
        self._configure_progress_bar(enable_progress_bar)

        # configure the ModelSummary callback
        # 注: 增加pytorch_lightning.callbacks.model_summary.ModelSummary
        self._configure_model_summary_callback(enable_model_summary)

        # accumulated grads
        # 注: 如果设置了梯度累累积, 则设置pytorch_lightning.callback.gradient_accumulation_scheduler.GradientAccumulationScheduler
        self._configure_accumulated_gradients(accumulate_grad_batches)

        # 注: ...
        if self.trainer.state._fault_tolerant_mode.is_enabled:
            self._configure_fault_tolerance_callbacks()

        self.trainer.callbacks.extend(_configure_external_callbacks())  # 注: 见下面的定义, 目前似乎是空列表

        # push all model checkpoint callbacks to the end
        # it is important that these are the last callbacks to run
        self.trainer.callbacks = self._reorder_callbacks(self.trainer.callbacks) # 注: 见下面的函数定义, Checkpoint被排放至最后

    @staticmethod
    def _reorder_callbacks(callbacks: List[Callback]) -> List[Callback]:
        tuner_callbacks: List[Callback] = []
        other_callbacks: List[Callback] = []
        checkpoint_callbacks: List[Callback] = []

        for cb in callbacks:
            if isinstance(cb, BatchSizeFinder):
                tuner_callbacks.append(cb)
            elif isinstance(cb, Checkpoint):
                checkpoint_callbacks.append(cb)
            else:
                other_callbacks.append(cb)

        return tuner_callbacks + other_callbacks + checkpoint_callbacks

def _configure_external_callbacks() -> List[Callback]:
    """Collect external callbacks registered through entry points.

    The entry points are expected to be functions returning a list of callbacks.

    Return:
        A list of all callbacks collected from external factories.
    """
    group = "pytorch_lightning.callbacks_factory"

    if _PYTHON_GREATER_EQUAL_3_8_0:
        from importlib.metadata import entry_points

        if _PYTHON_GREATER_EQUAL_3_10_0:
            factories = entry_points(group=group)  # type: ignore[call-arg]
        else:
            factories = entry_points().get(group, {})  # type: ignore[assignment]
    else:
        from pkg_resources import iter_entry_points

        factories = iter_entry_points(group)  # type: ignore[assignment]

    external_callbacks: List[Callback] = []
    for factory in factories:
        callback_factory = factory.load()
        callbacks_list: Union[List[Callback], Callback] = callback_factory()
        callbacks_list = [callbacks_list] if isinstance(callbacks_list, Callback) else callbacks_list
        _log.info(
            f"Adding {len(callbacks_list)} callbacks from entry point '{factory.name}':"
            f" {', '.join(type(cb).__name__ for cb in callbacks_list)}"
        )
        external_callbacks.extend(callbacks_list)
    return external_callbacks
```

理解上述代码需要如下Python知识

- Python包的EntryPoint的概念: 参考[知乎](https://zhuanlan.zhihu.com/p/37635578)`

**loop**

后面`trainer.fit`方法主要与这个`self.fit_loop`有关

```python
fit_loop = FitLoop(min_epochs=min_epochs, max_epochs=max_epochs)
training_epoch_loop = TrainingEpochLoop(min_steps=min_steps, max_steps=max_steps)
# 注: 执行的函数体为: fit_loop.epoch_loop=training_epoch_loop
fit_loop.connect(epoch_loop=training_epoch_loop)
# default .fit() loop
self.fit_loop = fit_loop
```

**self._call_callback_hooks**

`Trainer.fit` 方法中将会多次调用`Trainer._call_callback_hooks`方法, 因此有必要查看一下源码, 如下:

```python
class Trainer:
    def _call_callback_hooks(
        self,
        hook_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        log.debug(f"{self.__class__.__name__}: calling callback hook: {hook_name}")
        # TODO: remove if block in v1.8
        if hook_name in ("on_init_start", "on_init_end"):
            # these `Callback` hooks are the only ones that do not take a lightning module.
            # we also don't profile bc profiler hasn't been set yet
            for callback in self.callbacks:
                fn = getattr(callback, hook_name)
                if callable(fn):
                    fn(self, *args, **kwargs)
            return
        # 注: self.lightning_module的定义见如下注解
        pl_module = self.lightning_module
        if pl_module:
            prev_fx_name = pl_module._current_fx_name
            pl_module._current_fx_name = hook_name

        for callback in self.callbacks:
            fn = getattr(callback, hook_name)
            if callable(fn):
                with self.profiler.profile(f"[Callback]{callback.state_key}.{hook_name}"):
                    fn(self, self.lightning_module, *args, **kwargs)

        if pl_module:
            # restore current_fx when nested context
            pl_module._current_fx_name = prev_fx_name
```

注: 为何要对`"on_init_start", "on_init_end"`这两个做单独的处理? 因为其他的`hook_name`都在`Trainer.fit`方法内部被调用,`Trainer.fit`方法的源代码如下:
```python
class Trainer:
    def fit(
        self,
        model: "pl.LightningModule",
        train_dataloaders: Optional[Union[TRAIN_DATALOADERS, LightningDataModule]] = None,
        val_dataloaders: Optional[EVAL_DATALOADERS] = None,
        datamodule: Optional[LightningDataModule] = None,
        ckpt_path: Optional[str] = None,
    ) -> None:
        if not isinstance(model, pl.LightningModule):
            raise TypeError(f"`Trainer.fit()` requires a `LightningModule`, got: {model.__class__.__qualname__}")
        self.strategy._lightning_module = model
        call._call_and_handle_interrupt(
            self, self._fit_impl, model, train_dataloaders, val_dataloaders, datamodule, ckpt_path
        )
    @property
    def lightning_module(self) -> "pl.LightningModule":
        # TODO: this is actually an optional return
        return self.strategy.lightning_module

# src/pytorch_lightning/strategies/strategy.py
class Strategy(ABC):
    @property
    def lightning_module(self) -> Optional["pl.LightningModule"]:
        """Returns the pure LightningModule without potential wrappers."""
        return self._lightning_module
```


### `Trainer.fit`