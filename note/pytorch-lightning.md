
## 参考资料


示例代码:

- [pytorch-lightning 101](https://www.youtube.com/playlist?list=PLaMu-SDt_RB5NUm67hU2pdE75j6KaIOv2): 视频课程, 可以用来理解概念, 共 4 小节课程, 其中第 3 小节是 pytorch-lightning 的基本用法, 第 4 小节介绍了 pytorch-lightning 的实现细节

- [LightningLite](https://pytorch-lightning.readthedocs.io/en/latest/starter/lightning_lite.html): 轻量版的 pytorch-lighting, 目前(2022.9.29)并未完全成熟. 用于尽量做很少的代码改动, 快速将 pytorch 训练代码进行转换, 好处是可以很快地将写好的单 GPU 或 CPU 训练流程变得可以自动支持多卡训练, fp16 训练等.

- [pytorch-lightning with huggingface transfomers](https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/text-transformers.html)


## 使用


### pytorch_lightning.LightningModule


### pytorch_lightning.LightningDataModule

```python
from 
class MyDataset()

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
        self.datasets = {
            "train": MyDataset(...)
            "val": MyDataset(...)
        }

    def prepare_data(self):
        # 此部分代码仅在rank0上运行, 建议不要设置类的属性
        # 建议做一些数据的转换工作, 例如切分数据至train/val文件夹,将数据tokenizer化保存
        pass

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset["val"], batch_size=self.train_batch_size, shuffle=True)
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