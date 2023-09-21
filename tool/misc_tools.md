
格式化json输出
```bash
# 也可以制定别名
# alias prettyjson='python -m json.tool'
python -m json.tool a.json
```

# markdown 转 pdf

由于 Typora 现在收费，需要找一些提到方法：

**替代软件**

- [MarkText](https://github.com/marktext/marktext/releases)：可行

**VSCode 插件**

Markdown PDF：没有试验成功

**python 包**

一般做法是先将 markdown 转换为 html，而后再转为 pdf。下面的方法有些缺点待解决：

- `markdown` 将 markdown 字符串转换为 HTML 字符串时会出现例如 Markdown 语法中的代码块无法被转换正确的情况
- `pdfkit` 包用于将 HTML 字符串转化为 PDF，但对中文的支持还没找到较好的解决方案，并且这个包依赖于 wkhtmltopdf 这个软件


```bash
pip install markdown  # markdown -> html
pip install pdfkit
sudo apt-get install wkhtmltopdf
```

```python
import markdown
import pdfkit
with open('readme.md', 'r') as f:
    text = f.read()
html_text = markdown.markdown(text)  # 
pdfkit.from_string(html_text, 'out.pdf')
```

备注：目前还没搜索到符合以下条件的 Python 包：

- 直接用 `pip install` 进行安装，并且不需要额外用 `apt` 安装其他软件
- 安装完后提供命令行工具一键转换
- 默认对中文字符进行支持
- 提供对 Markdown 常见扩展语法的正常转换，并且不会丢失图片

# 内存及显存

参考自 [accelerate.utils.modeling.get_max_memory](https://github.com/huggingface/accelerate/blob/v0.19-release/src/accelerate/utils/modeling.py#L379)

```python
import torch
import psutil

_ = torch.tensor([0], device="cuda:0")
available, total = torch.cuda.mem_get_info(0)
print(f"{available/1024/1024}MB/{total/1024/1024}MB")

cpu_memory = psutil.virtual_memory()
total = cpu_memory.total
available = cpu_memory.available
print(f"{available/1024/1024}MB/{total/1024/1024}MB")
```

# http 免密

```
# ~/.netrc 适合 linux/MAC
machine: xx.com
login: username
password: xxxyyy
```


```
# ~/_netrc 适合 linux/MAC
machine: xx.com
login: username
password: xxxyyy
```

# 代理设置

参考[博客](https://solidspoon.xyz/2021/02/17/%E9%85%8D%E7%BD%AEWSL2%E4%BD%BF%E7%94%A8Windows%E4%BB%A3%E7%90%86%E4%B8%8A%E7%BD%91/)

前置条件: 使用 Windows 电脑并且使用 Clash 配置代理, 并打开 Allow LAN 选项, 下面介绍怎么在 Windows 上和 WSL2 上使用代理

情形一: 在 Windows 本机上, 在 python 中可以通过这种方式设置代理

```python
import os
# 假设在 Clash 中设置的 PORT 是 7890
os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890"
os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890"  # 注意这里的值不要带s
```

情形二: 在 WSL2 中

```bash
# ~/.bashrc
export hostip=$(cat /etc/resolv.conf |grep -oP '(?<=nameserver\ ).*')
export HTTP_PROXY="http://${hostip}:7890"
export HTTPS_PROXY="http://${hostip}:7890"
```

情形三: 在 Windows 本机上使用 `git clone` 命令:

**一次性使用**
```
git clone -c http.proxy="http://127.0.0.1:7890" https://github.com/huggingface/huggingface_hub.git
```

**永久配置**
```
git config --global http.https://github.com.proxy http://127.0.0.1:7890
git config --global https.https://github.com.proxy http://127.0.0.1:7890
```

**重置代理**
```
git config --global  --unset https.https://github.com.proxy
git config --global  --unset http.https://github.com.proxy
```

情形四: 在 huggingface datasets 库中使用 `proxies` 参数

```python
from datasets import load_dataset, DownloadConfig
# import os
# os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890"
# os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890"
load_dataset(
    'fka/awesome-chatgpt-prompts',
    DownloadConfig(proxies={"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"})
)
```

备注: 上述写法实际上会报错, 原因在于 `load_dataset` 方法会触发如下调用栈

```python
# datasets/load.py
from huggingface_hub import HfApi
hf_api = HfApi(config.HF_ENDPOINT)                  # config.HF_ENDPOINT: 'https://huggingface.co'
dataset_info = hf_api.dataset_info(
    repo_id=path,                                   # 'fka/awesome-chatgpt-prompts'
    revision=revision,                              # None
    use_auth_token=download_config.use_auth_token,  # None
    timeout=100.0,
)

# 进一步触发调用: 注意此处header中没有设置proxies, 而get_session中大概也没有配置proxies
# huggingface_hub/hf_api.py:HfApi.dataset_info
r = get_session().get(
    path,             # https://huggingface.co/api/datasets/fka/awesome-chatgpt-prompts
    headers=headers,  # {'user-agent': 'unknown/None; hf_hub/0.14.1; python/3.8.11; torch/1.11.0'}
    timeout=timeout,  # 100.0
    params=params     # {}
)

# huggingface_hub/utils/_http.py
# get_session() 最终基本会回落到这个函数
def _default_backend_factory() -> requests.Session:
    session = requests.Session()
    session.mount("http://", UniqueRequestIdAdapter())
    session.mount("https://", UniqueRequestIdAdapter())
    return session
```

