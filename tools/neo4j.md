# Neo4j

### 官方文档链接及笔记

**链接**

官方文档: [https://neo4j.com/docs/](https://neo4j.com/docs/), 其中`Neo4j DataBase`部分是对neo4j使用的文档, 包含图数据库的基本术语含义, 也Cyther语言的一些基本介绍, 因此入门阶段只需阅读其中的`Getting started`即可, `Cypher`部分是对Cyther语言的完整文档 面向开发者的文档: [https://neo4j.com/developer/](https://neo4j.com/developer/) lab文档?: [https://neo4j.com/labs/](https://neo4j.com/labs/) CQL官方手册: [https://neo4j.com/docs/cypher-refcard/current/](https://neo4j.com/docs/cypher-refcard/current/)

其他链接\(未整理\)

[Neo4j Sandbox - Get started quickly, no download required.](https://neo4j.com/sandbox/?ref=cypher) [Using Neo4j from Python - Developer Guides](https://neo4j.com/developer/python/) [Cypher Query Language - Developer Guides \(neo4j.com\)](https://neo4j.com/developer/cypher/)

### 安装和使用

首先去[官网](https://neo4j.com/download-center/)下载, 一般只需要下载Community Server版本即可, 注意Neo4j 4.x版本依赖JDK 11版本, Neo4j 3.x依赖JDK 8. 下载后解压放至某个目录即可, 例如: `D:\data\neo4j-community-4.2.5`

安装neo4j服务: [参考链接](https://my.oschina.net/u/3350450/blog/2245657)

```bash
cd D:\\data\\neo4j-community-4.2.5\\bin
# 安装neo4j服务, 只需一次即可
# 若报JDK版本不对的错误, 最直接的解决办法是在bin/neo4j.bat文件中增加类似如下一行
# 设置windows环境变量有时并不起作用
# SET "JAVA_HOME=C:\Program Files\Java\jdk-11.0.11"
neo4j.bat install-service
```

启动/关闭neo4j服务

```bash
bin/neo4j.bat start  # 启动neo4j服务
# 显示: Neo4j service started
bin/neo4j.bat stop # 停止neo4j服务
# 显示: Neo4j service stopped

bin/neo4j.bat restart  # 重启neo4j服务
bin/neo4j.bat status  # neo4j服务状态
```

访问方式1: cypher命令行

```bash
bin/cypher-shell.bat  # 启动cypher-shell命令行, 编写cyther语句与数据库交互
username: neo4j # 默认用户名为neo4j
password: ***** # 初始密码是neo4j
Password change required
new password: *** # cbx
Connected to Neo4j using Bolt protocol version 4.2 at neo4j://localhost:7687 as user neo4j.
Type :help for a list of available commands or :exit to exit the shell.
Note that Cypher queries must end with a semicolon.
# 进入cypher-shell命令行
neo4j@neo4j> :help # :exit表示退出, 或者ctrl+d快捷键也是退出
```

访问方式2: 浏览器访问\(也是第二种启动方式\)

```bash
bin/neo4j.bat console
# ... Bolt enabled on localhost:7687.
# ... Remote interface available at http://localhost:7474/

# ctrl+d退出
bin/neo4j.bat stop
```

### 基本概念

参考官方文档

| 术语 | 说明 |
| :--- | :--- |
| `Nodes` | 节点/实体 |
| `Labels` | 节点的标签, 一个节点可以有0个或多个标签 |
| `Relationships` | 关系, Neo4j里所有的关系都是有方向的, 但在数据库使用时可以根据需要忽略方向. 不推荐加上两个方向的边, 除非必须. 另外也允许自身到自身的边. |
| `Relationship types` | 每个关系都有且仅有一种关系类型 |
| `Properties` | 此概念同时适用于关系与实体 |
| `Traversals and paths` | 遍历与路径 |
| `Schema` | 代表indexes\(索引\)与constraints\(约束\), 可以导入数据之后根据需要增加或修改, 不用事先定好 |

命名是大小写敏感的, 约定命名风格如下

| 图数据库对象 | 建议风格 | 例子 |
| :--- | :--- | :--- |
| `Node labely` | 以大写开头的驼峰命名法 | `:VehicleOwner` |
| `Relationship type` | 全大写, 单词与单词之间用下划线连接 | `:OWNS_VEHICLE` |
| `Property` | 以小写开头的驼峰命名法 | `firstName` |

### 数据导入导出

#### 从CSV使用Cypher语句导入

从csv导入数据示例\(可参考[官网文档](https://neo4j.com/docs/getting-started/current/cypher-intro/load-csv/)\), 简述如下

首先准备3个csv文件如下, 放在import目录下并启动neo4j

```text
# persons.csv(用于建立人物实体)
id,name
1,Charlie Sheen
2,Michael Douglas
3,Martin Sheen
4,Morgan Freeman
# movies.csv(用于建立电影及国家实体, 以及电影与国家的关系)
id,title,country,year
1,Wall Street,USA,1987
2,The American President,USA,1995
3,The Shawshank Redemption,USA,1994
# roles.csv(用于建立人物与电影的关系)--注意导入时最后一条关系无法匹配上所以不会被导入
personId,movieId,role
1,1,Bud Fox
4,1,Carl Fox
3,1,Gordon Gekko
4,2,A.J. MacInerney
3,2,President Andrew Shepherd
5,3,Ellis Boyd 'Red' Redding
```

接下来依次输入如下cypher语句完成导入

```text
// 以下三条语句是建立主键, 索引等
CREATE CONSTRAINT personIdConstraint ON (person:Person) ASSERT person.id IS UNIQUE
CREATE CONSTRAINT movieIdConstraint ON (movie:Movie) ASSERT movie.id IS UNIQUE
CREATE INDEX FOR (c:Country) ON (c.name)

LOAD CSV WITH HEADERS FROM "file:///persons.csv" AS csvLine CREATE (p:Person {id: toInteger(csvLine.id), name: csvLine.name})

LOAD CSV WITH HEADERS FROM "file:///movies.csv" AS csvLine MERGE (country:Country {name: csvLine.country}) CREATE (movie:Movie {id: toInteger(csvLine.id), title: csvLine.title, year:toInteger(csvLine.year)}) CREATE (movie)-[:ORIGIN]->(country)

// 注意不能匹配到实体的行不会被导入(实体表中没有id为5的人物)
LOAD CSV WITH HEADERS FROM "file:///roles.csv" AS csvLine MATCH (person:Person {id: toInteger(csvLine.personId)}), (movie:Movie {id: toInteger(csvLine.movieId)}) CREATE (person)-[:ACTED_IN {role: csvLine.role}]->(movie)
```

所以总共导入了8个实体, 8个关系

```text
// 校验返回结果
// 7个实体(1个国家实体, 3个人物实体, 3个电影实体), 注意id为2的人物是孤立实体, 不在返回结果里
// 8个关系(5个ORIGIN关系, 3个ACTED_IN关系)
MATCH (n)-[r]->(m) RETURN n, r, m
```

**注意\(待确认\)**: 以上操作完后, 大概是会自动提交? 关闭neo4服务后重新启动, 导入的数据还在

#### 从CSV使用neo4j-admin导入

数据格式

```text
# 实体表表头
id:ID,name,year:int,:LABEL
# 关系表表头
:START_ID,:END_ID,:TYPE
```

导入语句

```bash
bin\\neo4j-admin import
--nodes person.csv
--relationships person_person.csv
--ignore-empty-strings true
--skip-duplicate-nodes true
--skip-bad-relationships true
--bad-tolerance 1500
--multiline-fields=true
```

#### 导出

将查询结果导出时似乎默认使用了`utf-8` **with BOM**的编码, 可能会造成一些麻烦

### CQL

所有笔记均来自于[官方文档](https://neo4j.com/docs/cypher-manual/current/).

#### 概念

**database与graph**

DBMS \(Database Management System\)：包含以及管理多个个database, 每个database下面有着多个graph。客户端程序会连接上DBMS并且针对这个DBMS打开一些session，一个session提供了DBMS中的任意graph的访问权限。

> A Neo4j Database Management System is capable of containing and managing multiple graphs contained in databases. Client applications will connect to the DBMS and open sessions against it. A client session provides access to any graph in the DBMS.

Graph：Graph是database的数据模型，通常情况下，一个database里只有一个Graph

Database：是硬盘或内存上的数据，并且包含数据检索机制

一般来说，Cypher语句是针对图的

备注: 网上找到关于data model，database model，database相关的概念，结合自己理解，未必正确大致意思如下是data model是逻辑上的概念，database model为具体的数据存储方案，database是实际上占用的内存或硬盘。例子：

data model：csv的列名以及每一列的数据类型（概念/逻辑约束）

database model：以逗号作为分割符进行数据域的划分（存储格式）

database：硬盘上的CSV文件（比特串）

#### 基本语句

* [x] MATCH
* [ ] OPTIONAL MATCH
* [x] RETURN
* [x] WITH
* [ ] UNWIND
* [x] WHERE
* [x] ORDER BY
* [x] SKIP
* [x] LIMIT
* [x] CREATE
* [x] DELETE
* [x] SET
* [x] REMOVE
* [ ] FOREACH
* [ ] MERGE
* [ ] CALL {} \(subquery\)
* [ ] CALL procedure
* [ ] UNION
* [x] USE
* [x] LOAD CSV

| 关键字 | 作用 |
| :--- | :--- |
| CREATE | 创建数据表, 节点, 关系, 属性 |
| MATCH | 检索有关的节点, 关系, 属性 |
| RETURN | 返回 |
| WHERE | 提供条件过滤 |
| DELETE | 删除节点和关系 |
| REMOVE | 删除节点或关系的属性 |
| ORDER BY | 排序 |
| SET | 添加或更新标签 |
| MERGE |  |

常用函数

labels\(n\) 节点n的标签列表 type\(r\) 关系的类型 id\(n\)/id\(r\) 返回节点或关系的id\(这里的id为Neo4j内部存储数据用的id, 具有全局唯一性\) n.attr/r.attr 返回节点或关系的

```text
/*列出所有数据库名*/
:dbs
/*选择某个数据库*/
:database_name
/*查看示例movie数据库实例*/
:play movie-graph

/**/
:help MATCH
```

**MATCH**

\(:NodeLabel {property\_key: property\_value}\)-\[:RelationLabel {k: v}\]-&gt;\(:Label {k:v}\)

```text
MATCH (n) RETURN n LIMIT 10;
MATCH (tom {name: "Tom Hanks"}) RETURN tom;

MATCH (a)-[:ACTED_IN]->(m)<-[:DIRECTED]-(d) RETURN a,m,d limit 300;

// MATCH不会反复经过同一个关系, 但可能会经过同一个节点
MATCH (tom:Person {name:"Tom Hanks"})-[:ACTED_IN]->(m)<-[:ACTED_IN]-(coActors) RETURN coActors.name
```

路径上的节点是否用变量储存, 节点标签/关系类型是否限定的写法

```text
// 实体使用变量,不加约束: (n)
// 关系不使用变量, 不加约束: -->
// 实体不使用变量, 不加约束: ()
// ...
(n)-->()-[r]-(n:Person|Event)-[:PARENT_IS|CHILDREN_IS]->(:Person)
```

路径

**WITH/AS**

**例子**

统计名字为三个重复字符的人且出度大于2的人的姓名及出度\(复杂写法\)

```text
MATCH (n:Person)-->(m:Person) WHERE n.name =~ "(.)\1{2}" WITH n, count(m) AS cnt WHERE cnt > 2 RETURN n.name, cnt
```

**RETURN/WHERE/ORDER BY/LIMIT/SKIP**

这几个语句是通用的, **次序不能颠倒**\(类似于MySQL\), 语句中出现的次序为:

WHERE - RETURN - ORDERBY - SKIP - LIMIT

WHERE用于筛选条件; RETURN用于确定返回哪些东西, 一般必不可少; ORDER BY用于将返回的结果排序; SKIP用于跳过返回的前几条数据\(一般不会单独用, 只会和ORDER BY连用\), LIMIT用于限定返回记录数.

备注: LIMIT限定时表示只返回前几条记录, 而不是随机挑选几条记录, 但如果不使用ORDER BY时, 看起来和随机挑选没什么两样.

**CREATE/DELETE**

create/delete用于创建/删除节点/关系

```text
// 创建节点, []内的东西是可选参数
CREATE (<node-name>:<label-name> [{<key>:<value>, <key>:<value>}] );
// 例子:
// CREATE (Carrie:Person {name:'Carrie-Anne Moss', born:1967})
// CREATE (:Movie {title:'The Matrix', released:1999, tagline:'Welcome to the Real World'})

// 删除节点基本语法, 也可以增加WHERE子句                                    
// 不加DETACH时, 只有孤立节点才能被删除, 加上DETACH后, 自动删除与该节点相关的关系
MATCH (n:Person {name: 'AAA'}) DELETE n
MATCH (n:Person {name: 'AAA'}) DETACH DELETE n
```

```text
// 创建关系
MATCH (a:Person), (b:Person) WHERE a.name = 'A' AND b.name = 'B' CREATE (a)-[r:RELTYPE {name: a.name+"->" + b.name}]->(b) RETURN type(r);
// 删除关系
MATCH (a:Person {name: "AAA"})-[r]-(b:Person {name: "BBB"}) DELETE r
```

create用于创建约束和索引

```text
// 创建约束
// 建立唯一性约束, personIdConstraint为约束名. 注意, 即使标签Person还不存在, 这条语句也是可以正常执行的.
CREATE CONSTRAINT personIdConstraint ON (person:Person) ASSERT person.id IS UNIQUE

// 建立索引
CREATE INDEX FOR (c:Country) ON (c.name)
```

**OPTIONAL MATCH**

**SET/REMOVE**

set/remove用于增加更新/删除节点或关系的属性/节点标签/关系类型

**USE**

```text
USE <graph> <other clauses>;
```

> Where `<graph>` refers to the name of a database in the DBMS.

#### 复杂语句练习

统计从某个实体出发, 两跳能到的节点以及到达这些节点的不同路径有多少条

```text
MATCH (n:Person {id: 'Per_471'})-[*1..2]->(x) RETURN labels(n), n.age, count(*), x.id
```

### python接口

[官网](https://neo4j.com/developer/python/)推荐了三个python模块, 如下

#### neo4j

```python
# pip3 install neo4j-driver
# python3 example.py

from neo4j import GraphDatabase, basic_auth

driver = GraphDatabase.driver(
  "bolt://<HOST>:<BOLTPORT>", 
  auth=basic_auth("<USERNAME>", "<PASSWORD>"))

cypher_query = '''
MATCH (p:Person {id:$ID}) RETURN p
'''

with driver.session(database="neo4j") as session:
  results = session.read_transaction(
    lambda tx: tx.run(cypher_query,
      ID="Per_1").data())

  for record in results:
    print(record)

driver.close()
```

#### py2neo

#### Neomodel

## 杂录

[野博客1\(也附带了一些基本py2neo操作\)](https://www.codenong.com/cs105494145/)

