
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

# 消息队列: RocketMQ

## 基本概念

RocketMQ 是阿里开发的消息队列组件, 涉及的一些概念参考[官方文档](https://rocketmq.apache.org/docs/introduction/02concepts)

- Producer: 生产者负责向队列里写消息
  - ProducerGroup: 每个生产者必须指定唯一的一个 ProducerGroup, 但每个队列都可以往队列发送不同的 Topic 的消息
- Consumer: 消费者负责从队列中读取消息, 注意一个消息被读取, 并不一定立即出队列
  - ConsumerGroup: 每个消费者必须指定唯一的一个 ConsumerGroup
  - Subscription: 消费者可以根据 Topic, Tag, Key 这些条件来过滤, 以获取感兴趣的消息. 注意一个消费者必须指定唯一的一个 Topic 进行订阅, 但同属一个 ConsumerGroup 的消费者可以指定不同的 Topic 进行订阅
  - ConsumerOffset: RocketMQ 会在内部为每个 ConsumerGroup 维护一个 Offset, 以避免消息被重复消费
- Topic:
- MessageQueue: 在一个 Topic 下的消息, RocketMQ 可能会使用多个队列来存储消息, 以实现负载均衡
- Message: 消息需要设置 Topic, 同一个 Topic 的消息会被负载均衡得放入至多个 MessageQueue 中, 每条 Message 可以设置 Tag 和 Key, 便于消费者根据这些元数据过滤信息
  - MessageTag: 
  - MessageKey:
  - MessageType: RocketMQ 支持 NORMAL, FIFO, TRANSACTION, DELAY 这几种消息类型 (TODO:这几个类型的具体含义)
  - MessageView: 消息的可读视图, 但不能修改消息
  - MessageOffset: 每个消息在进入消息队列时, RocketMQ 会记录这条消息所在 MessageQueue 的 Offset, Offset 的数据类型是 long int
  - MessageID: 作为消息的全局唯一标识符, ID 由 RocketMQ 内部自动设定, 确保消息在 RocketMQ 中能够被唯一识别.
  - MessageBody: 消息的实际内容, 可以是任意形式的字节数据
- TransactionChecker: 事务相关 (TODO: 具体含义)

Topic, Tag, Key 的通常用法: Topic 一般对应一个项目, Tag 对应于各个子项目, 而 Key 一般用于具体的业务逻辑约定. 例如: `Topic="SALE"` 表示销售场景的项目, 而 `Tag="ONLINE"` 和 `Tag="OFFLINE"` 分别代表线上和线下的场景, 而进一步用 `Key=buy_id` 用于表示一次具体的购买行为的唯一标识.



## 基础使用

启动服务, 完全参考 [https://rocketmq.apache.org/docs/quickStart/02quickstartWithDocker/](https://rocketmq.apache.org/docs/quickStart/02quickstartWithDocker/) 即可, 如下:

```bash
docker pull apache/rocketmq:5.3.0

# Start NameServer
docker run -d --name rmqnamesrv -p 9876:9876 --network rocketmq apache/rocketmq:5.3.0 sh mqnamesrv
# Verify if NameServer started successfully
docker logs -f rmqnamesrv

# Configure the broker's IP address
echo "brokerIP1=127.0.0.1" > broker.conf

# Start the Broker and Proxy
docker run -d --name rmqbroker --network rocketmq -p 10912:10912 -p 10911:10911 -p 10909:10909 -p 8080:8080 -p 8081:8081 -e "NAMESRV_ADDR=rmqnamesrv:9876" -v ./broker.conf:/home/rocketmq/rocketmq-5.3.0/conf/broker.conf apache/rocketmq:5.3.0 sh mqbroker --enable-proxy -c /home/rocketmq/rocketmq-5.3.0/conf/broker.conf

# Verify if Broker started successfully
docker exec -it rmqbroker bash -c "tail -n 10 /home/rocketmq/logs/rocketmqlogs/proxy.log"
```

使用消息队列(Python使用方式), 主要参考自 [https://github.com/apache/rocketmq-client-python/tree/master](https://github.com/apache/rocketmq-client-python/tree/master), 备注: 官方的 README 疑似有些地方没有及时和代码同步

```python
# pip install rocketmq==0.4.4 rocketmq-client-python==2.0.0
from rocketmq.client import Producer, Message
import time
from rocketmq.client import PushConsumer

topic_id = 'TID-YYY'

# 生产者
producer = Producer('PID-XXX')  # PID_XXX 是 group_id
# producer.set_name_server_address('127.0.0.1:9876')
producer.set_namesrv_addr('127.0.0.1:9876')
producer.start()
# 注意: producer 发送的 topic_id 可以各不相同
msg = Message(topic_id)
msg.set_keys('X1')
msg.set_tags('X2')
msg.set_body('hello')
ret = producer.send_sync(msg)
print(ret.status, ret.msg_id, ret.offset)
producer.shutdown()


# 消费者
def callback(msg):
    print(type(msg), msg.id, msg.body, msg.tags, msg.keys, msg.topic)
consumer = PushConsumer('CID_XXX')  # CID_XXX 是 group_id
# consumer.set_name_server_address('127.0.0.1:9876')
consumer.set_namesrv_addr('127.0.0.1:9876')
# 注意: consumer 订阅时必须指定 topic, 其他筛选条件均为可选
consumer.subscribe(topic_id, callback)
consumer.start()

while True:
    time.sleep(3600)
consumer.shutdown()
```

输出:

```
SendStatus.OK 0101007F0000A61200003CF1C4030100 3
<class 'rocketmq.client.RecvMessage'> 0101007F0000A61200003CF1C4030100 b'hello' b'X2' b'X1' TID-YYY
```

## 进阶使用

**样例1: 鉴权**

```python
from rocketmq.client import PushConsumer, MessageModel
consumer = PushConsumer(group_id="CID_XXX", orderly=False, message_model=MessageModel.CLUSTERING)  # 默认值
# orderly 可以设置为 True, message_model 还可以设置为 MessageModel.BROADCASTING
consumer.set_session_credential(access_key="aa", access_secret="aa", channel="FMQ")
```

与 手机APP 类比: 其中 `access_key` 相当于用户名, `access_secret` 相当于密码, `channel` 相当于设备平台类型, 例如: iOS/Android

**样例2(待完善)**

`producer.py`:

```python
from rocketmq.client import Producer, Message
import time
from rocketmq.client import PushConsumer

topic_id = 'TID-YYY'

producer = Producer(group_id='PID-XXX')
producer.set_namesrv_addr('127.0.0.1:9876')
producer.start()

while True:
    msg = Message(topic_id)
    key = input("请输入key:")
    msg.set_keys(key)
    tag = input("请输入tag:")
    msg.set_tags(tag)
    body = input("请输入body:")
    msg.set_body(body)
    ret = producer.send_sync(msg)
    print("发送消息成功: ", ret)
producer.shutdown()
```

`consumer.py`:

疑问:
- `keys` 里面各项的含义
- `PushConsumer` 里 `orderly` 和 `message_model` 的含义
- `subscribe` 方法 `expression` 参数应该怎么填
- producer/consumer 的 group-id 的含义

```python
from rocketmq.client import Producer, PushConsumer, Message, MessageModel
import time

topic_id = 'TID-YYY'

def callback(msg):
    # type(msg): rocketmq.client.RecvMessage
    keys = [
        "topic",
        "tags",
        "keys",
        "id",
        "body",
        "queue_id",
        "queue_offset",
        "commit_log_offset",
        "prepared_transaction_offset",
        "delay_time_level",
        "reconsume_times",
        "born_timestamp",
        "store_timestamp",
        "store_size"
    ]
    print(">>> "+", ".join([f"{key}: {getattr(msg, key)}" for key in keys]))

consumer = PushConsumer(group_id='CID_XXX', orderly=False, message_model=MessageModel.CLUSTERING)
consumer.set_namesrv_addr('127.0.0.1:9876')
consumer.subscribe(
    topic=topic_id,
    callback=callback,
    expression="*",
)
consumer.start()

while True:
    time.sleep(3600)
consumer.shutdown()
```

**例子3**

(1) 使用 fastapi 实现服务端

接口入参:

```json
{
    "question": "talk a story"
}
```

接口出参:

```json
{
    "requestId": "key-123"
}
```

接口在返回后, 会作为消费者向 rocketMQ 里写消息多个消息, 且 topic 固定为 "TID-1", key 为接口出参里的 requestId, 消息内容是这样的(首先流式返回文本, 用 message表示, 结束用end标识)

```
{"answer": "There is", "event": "message"}
{"answer": " a", "event": "message"}
{"answer": " desktop.", "event": "message"}
{"answer": "", "event": "end"}
```

(2) 使用 requests 实现 python 客户端

(3) h5 实现客户端

## 杂录

PushConsumer 与 PullConsumer 的区别