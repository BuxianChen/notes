## 整体代码结构

主要的基类如下

- `PreTrainedModel`: 模型
- `PretrainedConfig`: 配置
- `PreTrainedTokenizerBase`: tokenizer

在以上三个类之上, `Pipeline` 基类用于组合这三个类.

另外, 还有些小东西: `ModelOutput`, 是模型输出的结果的基类.

`transformers` 的代码总体遵循的[设计哲学](https://huggingface.co/docs/transformers/v4.22.1/en/philosophy)是不强调代码复用, 比如没有一个 `attention.py` 文件中实现所有的注意力机制, 与之相对应的是将所有的模型基本上写在三个文件里, 例如在 `transformers/models/bart` 文件夹里与 `pytorch` 有关的代码文件如下:

```
modeling_bart.py
configuration_bart.py
tokenizer_bart.py
```


## PreTrainedModel

### 使用

```python
pretrained_model_name_or_path = "fnlp/bart-base-chinese"
from transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
model = BartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path)
text2text_generator = Text2TextGenerationPipeline(model, tokenizer)  
text2text_generator("北京是[MASK]的首都", max_length=50, do_sample=False)
# output:
# [{'generated_text': '北 京 是 中 国 的 首 都'}]
```

### 源码解析

transformers 代码中的带有 `from_pretrained` 的类都继承自 `PreTrainedModel`, 其具体继承关系如下:

```python
class ModuleUtilsMixin:
    pass
    # (@staticmethod) def _hook_rss_memory_pre_forward
    # (@staticmethod) def _hook_rss_memory_post_forward
    # def add_memory_hooks
    # def reset_memory_hooks_state
    # def invert_attention_mask
    # def get_extended_attention_mask
    # def get_head_mask
    # def _convert_head_mask_to_5d
    # def num_parameters
    # def estimate_tokens
    # def floating_point_ops
    # (@property) device
    # (@property) dtype

class GenerationMixin:
    pass
    # ===========
    # public methods:
    # ===========
    # @torch.no_grad()
    # def generate
    # def greedy_search
    # def sample
    # def beam_search
    # def beam_sample
    # def group_beam_search

class PushToHubMixin
    pass
    # ===========
    # public methods:
    # ===========
    # def push_to_hub

class PreTrainedModel(nn.Module, ModuleUtilsMixin, GenerationMixin, PushToHubMixin):
    # ...
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        **kwargs):

        ...
        model = cls(config, *model_args, **model_kwargs)
        ...
        return model
        
    # def save_pretrained
```

### example: BartForConditionalGeneration

具体到上面的例子中:

```python
# transformers/models/bart/modeling_bart.py
class BartPretrainedModel(PreTrainedModel):
    # some class attributes, ...
    def _init_weights(self, module):
        # pass ...
    def _set_gradient_checkpointing(self, module, value=False):
        # pass ...

class BartModel(BartPretrainedModel):
    # 具体的模型定义...
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)  # 继承自BartPretrainedModel
        self.decoder = BartDecoder(config, self.shared)  # 继承自BartPretrainedModel

        # Initialize weights and apply final processing
        self.post_init()
    def forward(self, input_ids, attention_mask, ..., labels, ...):
        # pass ...
        # returns: Seq2SeqModelOutput

class BartForConditionalGeneration(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        # Initialize weights and apply final processing
        self.post_init()
    def forward(self, input_ids, attention_mask, ..., labels, ...):
        # pass
        # returns: Seq2SeqModelOutput
```

### ModelOutput

```python
class ModelOutput(OrderedDict):
    def __post_init__(self): ...
    def __delitem__(self, *args, **kwargs): ...
    def setdefault(self, *args, **kwargs): ...
    def pop(self, *args, **kwargs): ...
    def update(self, *args, **kwargs): ...
    def __getitem__(self, k): ...
    def __setattr__(self, name, value): ...
    def __setitem__(self, name, value): ...
    def to_tuple(self): ...

@dataclass
class Seq2SeqModelOutput(ModelOutput):
    # some attrs
```

## Pipeline

### 使用

```python
from transformers import pipeline
generator = pipeline(task="text-generation")
generator(
    "Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone"
)  # doctest: +SKIP

# outputs: [{'generated_text': 'Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone, Seven for the Iron-priests at the door to the east, and thirteen for the Lord Kings at the end of the mountain'}]
```


### 源码解析

```python
class _ScikitCompat(ABC):

    @abstractmethod
    def transform(self, X):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError()

class Pipeline(_ScikitCompat):
    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        tokenizer: Optional[PreTrainedTokenizer] = None,
        feature_extractor: Optional[PreTrainedFeatureExtractor] = None,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        task: str = "",
        args_parser: ArgumentHandler = None,
        device: int = -1,
        binary_output: bool = False,
        **kwargs,
    ):

# src/transformers/pipelines/__init__.py
def pipeline(...):
    ...
    pipeline_class(model=model, framework=framework, task=task, **kwargs)
```

`framework` 取值为 `tf` 或者 `pt`, 代表 `tensorflow` 和 `pytorch`. 一般用于指示代码的输出为 `tf.tensor` 或 `torch.tensor`. 