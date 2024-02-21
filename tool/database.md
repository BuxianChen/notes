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
