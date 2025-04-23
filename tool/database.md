# sqlite

## Python Client

**查看所有的表结构信息**

```python
import sqlite3

conn = sqlite3.connect('your_database_name.db')
cursor = conn.cursor()

# 获取数据库中所有的表格名
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

# 遍历每个表格，打印其schema, 并打印数据表里的数据
for table in tables:
    print("="*50)
    table_name = table[0]
    print(f"Schema for table: {table_name}")
    cursor.execute("PRAGMA table_info({})".format(table_name))
    schema = cursor.fetchall()
    for column in schema:
        print(column[1], column[2], end=", ")
    print("\n")
    print(f"Data for table: {table_name}")
    cursor.execute(f"select * from {table_name}")
    result = cursor.fetchall()
    for row in result:
        print(row)
    print("\n")

# 关闭连接
conn.close()
```

## DB Browser for SQLite

下载地址: [https://sqlitebrowser.org/](https://sqlitebrowser.org/)

可以交互式地查看修改 sqlite 的 `xx.db` 文件

坑: 在 windows 上安装 x64 版本的 DB Browser for SQLite, 但希望 `xx.db` 位于 WSL2 内的目录, 无法达到目的.

# MySQL

## 部署: Docker (TODO)

## Python Client (TODO)

# sqlalchemy

sqlalchemy 有两套 API, 一套被称为 Core, 另一套被称为 ORM, 分别使用如下方式进行导入:

```python
from sqlalchemy import xxx      # Core API
from sqlalchemy.orm import xxx  # ORM API
```

其中 ORM 是对 Core 的上层封装, 也更 pythonic

## 建立连接

连接到数据库需要先建立 engine (连接池)

```python
from sqlalchemy import create_engine
# sqlite 数据库, 使用 pysqlite 包 (默认情况用 sqlite3 内置包), 在内存中建立数据库
engine = create_engine("sqlite+pysqlite:///:memory:", echo=True)  # 在执行 sql 语句时会打印语句
engine = create_engine("sqlite:///example.db")
# mysql 数据库, 使用 pymysql 包: 登录的用户名, 密码, 服务器地址及端口号, 数据库名
engine = create_engine("mysql+pymysql://{user}:{password}@{host}:{port}/{database_name}")


engine.dispose()  # 释放连接池
```

### Core: connect

从连接池里获取一个连接, 以进行数据库操作

```python
connect = engine.connect()
connect.close()  # 关闭连接

# 或者使用 with 语法
with engine.connect() as connect:  # __exit__ 时会触发 close
    ...
```

### ORM: session

使用 ORM API 时, 与 connect 对应的概念是 Session

```python
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker

# 方式 1
session = Session(engine)

# 方式 2
Session = sessionmaker(bind=engine)
session = Session()

session.close()  # 关闭连接

# 或者使用 with 语法
with Session(engine) as session:  # __exit__ 时会触发 close
    ...
```

## 构建表

### Core 构建表

使用 Core 的方式定义 Table 的方式被称为 Table constructor. 一般都是用多个 `Table` 共享一个 `MetaData` 实例.

```python
from sqlalchemy import Table, Column, Integer, String
from sqlalchemy import MetaData
from sqlalchemy import ForeignKey

metadata_obj = MetaData()  # sqlalchemy.sql.schema.MetaData

user_table = Table(
    "user_account",  # 数据库里的表名
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("name", String(30)),
    Column("fullname", String),
)

address_table = Table(
    "address",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("user_id", ForeignKey("user_account.id"), nullable=False),
    Column("email_address", String, nullable=False),
)

metadata_obj.create_all(engine)
```

### ORM 构建表

使用 ORM 的方式定义 Table 的方式被称为: ORM Mapped classes / Declarative Forms, 具体如下:

Step 1: Base,可以用任意一种方式进行

```python
# 方法 1: SQLAlchemy 2.0 推荐的方式
from sqlalchemy.orm import DeclarativeBase
class Base(DeclarativeBase):
    pass

# 方法 2
from sqlalchemy.orm import declarative_base
Base = declarative_base()

Base.metadata   # sqlalchemy.sql.schema.MetaData 对象, 与 Core API 里显式用 metadata_obj = MetaData() 得到的 metadata_obj 对应
```

Step 2: 继承 Base

```python
from typing import List
from typing import Optional
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy import Column
from sqlalchemy.orm import relationship

class User(Base):
    __tablename__ = "user_account"  # 数据库里的表名
    id: Mapped[int] = mapped_column(primary_key=True)
    # id = Column(Integer, primary_key=True)  # 这种写法是类似于 Core API 的写法, 许多网上的资料会用这种写法 
    name: Mapped[str] = mapped_column(String(30))
    fullname: Mapped[Optional[str]]
    addresses: Mapped[List["Address"]] = relationship(back_populates="user")  # 不存储在数据库里
    def __repr__(self) -> str:
        return f"User(id={self.id!r}, name={self.name!r}, fullname={self.fullname!r})"

class Address(Base):
    __tablename__ = "address"
    id: Mapped[int] = mapped_column(primary_key=True)
    email_address: Mapped[str]
    user_id = mapped_column(ForeignKey("user_account.id"))  # 定义外键约束, 数据库层面的约束
    user: Mapped[User] = relationship(back_populates="addresses")
    def __repr__(self) -> str:
        return f"Address(id={self.id!r}, email_address={self.email_address!r}, user_id={self.user_id!r})"
```

relationship 是可选的, 实际的数据库存储里并不包含 `user` 及 `addresses` 这两列, 它与 `ForeignKey` 的关系以及给 ORM API 带来的便利性具体见后续, 大体如下(TODO: 待确认)

```python
# 假设 user 是一个 User 对象, 在 User 里定义了 addresses = relationship(back_populates="user")
user.addresses[0].email_address

from sqlalchemy.orm import Session

with Session(engine) as session:
    sandy = session.query(User).filter_by(name="sandy").first()
    if sandy:
        for addr in sandy.addresses:
            print(addr.email_address)
```

## 操作表: 增删改查

TODO: 下例其实有些杂糅, 操作语句用的 Core API, 而执行用的 ORM API, 本质上此例算作 Core API

```python
from sqlalchemy import create_engine
from sqlalchemy import select, text
from sqlalchemy.orm import sessionmaker
from sql_test import Address, User

engine = create_engine('sqlite:///example.db', echo=False)
Session = sessionmaker(bind=engine)
with Session() as session:
    result = session.execute(text("SELECT * FROM user_account"))
    for row in result:
        print(type(row), row)  # (sqlalchemy.engine.row.Row, (1, 'Ask', 'AskBob'))
    result1 = session.execute(select(User))
    for row1 in result1:
        print(type(row), row1)  # (sqlalchemy.engine.row.Row, (User(id=1, name='Ask', fullname='AskBob'),))

    # 两种查询方式结果有所不同
    # for item in result.mappings(): print(item)
    # {'id': 1, 'name': 'Ask', 'fullname': 'AskBob'}
    # for item in result1.mappings(): print(item)
    # {'User': User(id=1, name='Ask', fullname='AskBob')}
```

### Core

```python
from sqlalchemy import select, text

stmt1 = text("select * from xx")  # stmt1 是一个 TextClause 对象, sqlalchemy.sql.elements.TextClause
stmt2 = select(User)  # stmt2 是一个 Select 对象 (sqlalchemy.sql.selectable.Select)

# 打印 SQL 语句
print(stmt1)  # "select * from xx"
print(stmt2)  # "SELECT user_account.id, user_account.name, user_account.fullname FROM user_account"

# 备注: stmt2 转化为 SQL 语句字符串的过程会先经过 compile
# compiled = stmt2.compile()

result = connect.execute(stmt2)
```

### ORM

## 杂记

```python
from sqlalchemy import select
from sqlalchemy.orm import query

# 注意 select(User) 只是语句, 可以使用 print(select(User)) 打印对应的语句
row = session.execute(select(User)).first()[0]  # User(...)
row = session.scalars(select(User)).first()     # User(...)

# 注意 query(User) 只是语句, 可以使用 print(query(User)) 打印对应的语句 
rows = session.query(User).all()                # [User(...), User(...)]
# 注意 query(User).filter(User.id==1) 只是语句
rows1 = session.query(User).filter(User.id==1).all()  # [User(...)]

user = session.get(User, 1)  # 直接执行, user=User(...)
user_alias = session.get(User, 1)

user is user_alias  # True, 在当前会话里, 只会存一个副本
user is rows[0]     # True, 在当前会话里, 只会存一个副本
```

# faiss

faiss 只支持稠密向量的 IP(内积) 和 L2 距离

# neo4j

## Server

要启用 apoc (`langchain`, `llama_index` 等一般都需要这个功能) 需要对官方镜像打上补丁:

```Dockerfile
FROM neo4j:4.4.29
RUN cp /var/lib/neo4j/labs/apoc-* /var/lib/neo4j/plugins
```

build 镜像

```bash
docker build -t neo4j:4.4.29.patch .
```

启动镜像 (可以多个版本独立部署):

```python
mkdir -p $HOME/temp/neo4j-data-4.4.29/data && sudo mkdir -p $HOME/temp/neo4j-data-4.4.29/logs && sudo chmod -R 777 temp/neo4j-data-4.4.29

# https://github.com/langchain-ai/langchain/issues/12901
docker run -it --rm -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/12345678 -v $HOME/temp/neo4j-data-4.4.29/data:/data -v $HOME/temp/neo4j-data-4.4.29/logs:/logs -e NEO4J_dbms_security_procedures_unrestricted=apoc.* -e NEO4J_dbms_security_procedures_allowlist=apoc.* neo4j:4.4.29.patch
```

注意:

- 部署多个容器时, 使用 Neo4j browser 登录时要注意选择好正确的 bolt 地址
- neo4j 社区版一个实例只能有一个图 (database)

## Python Client (TODO)


# Milvus 2.3.x

使用 Docker 启动服务, 并安装相应的 Python Client

```bash
wget https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh
bash standalone_embed.sh start
pip install protobuf grpcio-tools pymilvus
```

- database: 默认是 `default`, 对应于关系型数据库管理系统 (RDBMS) 中的 database
- collection: 对应于 RDBMS 中的 table
- partition: 

```python
from pymilvus import connections
from pymilvus import Collection
from pymilvus import list_collections
from pymilvus import db
from pymilvus import utility

# 1. 连接
connections.connect(
    alias="myconnect", host="127.0.0.1", port='19530',
)
connections.disconnect("myconnect")
# MilvusClient()  # 另一套 API

# 2. database
db.list_database()  # ["default"]
database = db.create_database("book")
db.using_database("book")
db.drop_database("book")

# 3. collection, schema, index
collection = Collection("LangChainCollection")
collection.create_index(field_name="vector", index_name="myindex")
utility.list_indexes("LangChainCollection")  # ["myindex"], 索引名

collection.load()
collection.indexes  # 所有索引
collection.indexes[0].field_name  # 索引建立在哪个字段上, "vector"

data = [{"vector": [0.1, -0.2], "text": "ss", "source": "A"}]
collection.insert(data)
collection.flush()  # 使插入数据生效
collection.num_entities  # collection 的行数

collection.release()  # 修改索引需要先对 collection 释放
collection.drop_index(index_name="myindex")  # 根据索引名删除

# 4. 向量相似度查询 与 根据字段过滤 混合检索

# search: 按向量相似度
import random
queries = [[random.random() for i in range(1024)] for k in range(2)]
search_params = {
    "metric_type": "L2",
    # "offset": 0,
    # "ignore_growing": False,
    "params": {"nprobe": 10}
}

results = collection.serach(
    data=queries,
    anns_field="vector",  # 指定用来做向量相似度查询应用的字段
    params=search_params,
    limit=3,  # 每个查询返回几条查询结果
    expr=None,  # 按字段进行过滤的语句
    output_fields=["source", "text", "pk", "vector"],  # 返回字段名, 这里假设pk是主键
    consistency_level="Strong"
)

assert len(result) == 2
assert len(result[0]) == 3
assert result[0][0].id == result[0][0].fields['pk']
result[0][0].distance  # 相似度
result[0][0].fields: Dict  # {"source": xxx, "text": xxx, "pk": xxx, "vector": [0.1, ..., -0.3]}
```

# Milvus 2.4.x 新特性

新特性说明可参考 release 信息: [https://github.com/milvus-io/milvus/releases/tag/v2.4.0-rc.1](https://github.com/milvus-io/milvus/releases/tag/v2.4.0-rc.1)

## Embedding

- 文档: [https://milvus.io/docs/embeddings.md](https://milvus.io/docs/embeddings.md)

以下用法来自上述官方文档

```python
# pip install pymilvus[model]
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]
query = "Who started AI research?"
bge_m3_ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")

docs_embeddings = bge_m3_ef(docs)
query_embeddings = bge_m3_ef([query])
```

备注: 实质上这里是对 FlagEmbedding 的简单封装, 因此算不上是 Milvus 的特性

## sparse vector

- 关于 sparse vector search: [https://milvus.io/docs/sparse_vector.md](https://milvus.io/docs/sparse_vector.md)

备注: 关于 sparse vector 所使用的距离函数目前只支持内积 (IP), 不支持 L2 及 COSINE.

## hybrid_search

- API 文档: [https://milvus.io/api-reference/pymilvus/v2.4.x/ORM/Collection/hybrid_search.md](https://milvus.io/api-reference/pymilvus/v2.4.x/ORM/Collection/hybrid_search.md
)
- 例子: [https://github.com/milvus-io/pymilvus/blob/master/examples/hello_hybrid_sparse_dense.py](https://github.com/milvus-io/pymilvus/blob/master/examples/hello_hybrid_sparse_dense.py)

备注: 所谓的 `hybrid_search`, 实质上只是根据多个 vector 类型的字段进行独立召回, 然后再对多个召回结果 rerank, 而目前 milvus 支持加权与 RRF 的 rerank 方法. 因此实质上 `hybrid_search` 也算不上是新特性.

## fuzzy match: prefixes, infixes, suffixes search

似乎之前版本的 Milvus 只支持前缀索引 (`text like 'the%'`), 2.4 之后支持前缀, 中缀, 后缀索引. 以下代码参考自 [https://github.com/milvus-io/pymilvus/blob/2.4/examples/fuzzy_match.py](https://github.com/milvus-io/pymilvus/blob/2.4/examples/fuzzy_match.py)

```python
res = collection.query(expr='title like "The%"', output_fields=["id", "title"])
res = collection.query(expr='title like "%the%"', output_fields=["id", "title"])
res = collection.query(expr='title like "%Rye"', output_fields=["id", "title"])
res = collection.query(expr='title like "Flip_ed"', output_fields=["id", "title"]) # _ 代表一个任意字符
# you can create inverted index to accelerate the fuzzy match.
collection.release()
collection.create_index(
    "title", {"index_type": "INVERTED"})
collection.load()
```

## Grouping Search

也就是对 multi-vector retriever 的支持 (特别地: 可用于 ParentDocumentRetriever), 代码参考 [https://github.com/milvus-io/pymilvus/blob/2.4/examples/example_group_by.py](https://github.com/milvus-io/pymilvus/blob/2.4/examples/example_group_by.py)

```python
# 这里 vectors[:3] 是 3 个向量: List[List[float]]
# doc_id 字段只有 44 个取值 (可以理解为 44 篇文章), batch_size=100
result = collection.search(vectors[:3], "float_vector", search_params, limit=batch_size, timeout=600,
                           output_fields=["chunk_id", "doc_id"], group_by_field="doc_id")
# 最终 result 里对于每个检索向量, 最终地检索结果只有 44 个 (小于预定义的 100)
# 实际的运作逻辑是: 按 doc_id 分组, 以同一个 doc_id 的所有向量里最相似的分数作为这个 doc_id 的相似度值, 然后排序 (注意返回结果里每个 doc_id 底下只返回最相似的那个 chunk)
```

备注: 这个特性仍然无法用作支持 BGE-M3 的 colbert 向量搜索

## MilvusClient

在 python client 方面, 将 MilvusClient 这种用法做了进一步完善, 估计后续版本的主流用法应该会是用 MilvusClient.

# Weaviate

Weaviate 是一个向量数据库, 支持混合检索. 根据下面的文章可以看出, 其实际上只是分别检索, 然后 rerank 实现的. 注意字面检索使用的是 BM25/BM25F, 而 rerank 可以选择加权重或者是 RRF. 官方比较推荐用加权重的方式 rerank.

这里简述下运作逻辑: 假设最终需要检索 k 个文本, 那么分别用字面检索和向量检索得到 k 个文本 (目前似乎不能设置为多于 k 个, 或者其内部有可能设置更高, 但似乎不对用户暴露), 当使用加权 rerank 时, 首先分别将字面检索/向量检索的分数按线性变换到 0-1 之间, 即最相似的文本的相似度为 1, 第 k 个文本的相似度为 0. 然后再加权重 (权重可以设置), 最后排序得到最终的 k 个文本.

- 关于 hybrid search 的具体运作逻辑: [https://weaviate.io/blog/hybrid-search-fusion-algorithms](https://weaviate.io/blog/hybrid-search-fusion-algorithms)

# Redis

## Docker 运行

```bash
docker run -d --name redis-test -p 6379:6379 redis:latest
```

## 发布/订阅模式

**发布者**

```python
# redis_publisher.py
import redis.asyncio as redis
import json
import asyncio

async def publish_main():
    client = redis.Redis(
        host="127.0.0.1",
        port=6379,
        password=None,
        decode_responses=True
    )
    channel_name = "test"
    await client.ping()

    while True:
        message = input("User: ")
        if message.lower() == "exit":
            print("Exiting...")
            break
        # 发送消息需要指定发布到哪个 channel, 可以发送字符串
        await client.publish(channel_name, message)

if __name__ == "__main__":
    asyncio.run(publish_main())
```


**订阅者**

```python
# redis_subscriber.py
import redis.asyncio as redis
import asyncio

async def process_fn(*args, **kwargs):
    print(args, kwargs)

async def subscribe_main():
    client = redis.Redis(
        host="127.0.0.1",
        port=6379,
        password=None,
        decode_responses=True
    )
    channel_name = "test"

    await client.ping()
    pubsub = client.pubsub()
    await pubsub.subscribe(channel_name)
    try:
        # 不推荐:
        # get_message 是非阻塞方法, 总会立刻返回, 但如果没有获取到消息, 将返回 None, 因此采用 while True 轮询的方式
        # while True:
        #     message = await pubsub.get_message(ignore_subscribe_messages=True)
        #     if message:
        #         await process_fn(message)

        # 推荐:
        # listen 是阻塞方法, 获取到消息时才会继续执行 async for 里面的内容
        async for message in pubsub.listen():
            if message['type'] == 'message':
                await process_fn(message)
    finally:
        await pubsub.unsubscribe(channel_name)
        await pubsub.close()

if __name__ == "__main__":
    asyncio.run(subscribe_main())
```

运行方式: 打开两个终端, 一个运行 `python redis_publisher.py`, 另一个运行 `python redis_subscriber.py`

```bash
# redis_publisher.py 的终端交互
User: 123
User: 234
User: exit
Exiting...

# redis_subscriber.py 的终端输出
({'type': 'message', 'pattern': None, 'channel': 'test', 'data': '123'},) {}
({'type': 'message', 'pattern': None, 'channel': 'test', 'data': '234'},) {}
```

Redis 的发布/订阅一般不会存储消息, 也就是说假设先启动 `redis_publisher.py` 并且先发送了一条消息, 然后再启动 `redis_subscriber.py`, 那么这一条消息将不会被收到. 上面的例子是异步的写法.
