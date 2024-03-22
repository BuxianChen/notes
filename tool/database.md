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

# sqlalchemy


## 构建表

### Core 构建表

一般都是用多个 `Table` 共享一个 `MetaData`

```python
# 方式一: Core: Table constructor
from sqlalchemy import Table, Column, Integer, String
from sqlalchemy import MetaData
from sqlalchemy import ForeignKey

metadata_obj = MetaData()  # sqlalchemy.sql.schema.MetaData

user_table = Table(
    "user_account",
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

```python
# 方式二: ORM: ORM Mapped classes/Declarative Forms

# step 1: Base, 可以用任意一种方式进行
from sqlalchemy.orm import DeclarativeBase
class Base(DeclarativeBase):
    pass
# Base.metadata   # sqlalchemy.sql.schema.MetaData 对象

# from sqlalchemy.orm import declarative_base
# Base = declarative_base()

# step 2:
from typing import List
from typing import Optional
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

class User(Base):
    __tablename__ = "user_account"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(30))
    fullname: Mapped[Optional[str]]
    addresses: Mapped[List["Address"]] = relationship(back_populates="user")
    def __repr__(self) -> str:
        return f"User(id={self.id!r}, name={self.name!r}, fullname={self.fullname!r})"

class Address(Base):
    __tablename__ = "address"
    id: Mapped[int] = mapped_column(primary_key=True)
    email_address: Mapped[str]
    user_id = mapped_column(ForeignKey("user_account.id"))
    user: Mapped[User] = relationship(back_populates="addresses")
    def __repr__(self) -> str:
        return f"Address(id={self.id!r}, email_address={self.email_address!r})"
```

## 操作表

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

# Milvus

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
