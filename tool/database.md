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

# 遍历每个表格，打印其schema
for table in tables:
    table_name = table[0]
    print("Schema for table:", table_name)
    cursor.execute("PRAGMA table_info({})".format(table_name))
    schema = cursor.fetchall()
    for column in schema:
        print(column[1], column[2], end=", ")
    print("\n")

# 关闭连接
conn.close()
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
