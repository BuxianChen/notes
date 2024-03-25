# SQL

子查询与子句是不同的概念: 子句是值 group by 子句, where 子句等

子查询要带括号, 但无需 `AS`，例如：

```sql
select a.b from
    (select b, c from tablea) a
where a.c > '2020'
```

子查询按查询结果分:

- 标量
- 一行
- 一列
- 一张表 (多行多列)


partition by 要注意必须嵌套, hive 中的随机数用 RAND, 例如以下表示按 country 和 city 进行分组, 每组随机取 3 个名字

```sql
select country, city, name, rn from (
    select country, city, name, ROW_NUMBER() over (PARTITION BY country, city ORDER BY RAND() ) AS rn
) d
where d.rn < 3
```

注意有些表可能是月表, 可能就没有满足条件 `xx_date > '2020' and xx_date < '2021'` 的数据

函数测试方法
```sql
-- 注意字符串一般都是用单引号'
select datediff('2023-04')
```

以下章节标题已定义清楚

## 查: SELECT

## 增改删: INSERT, UPDATE, DELETE

```sql
-- 允许插入部分列, 但只能插入一行数据
INSERT INTO table_name(col1, col2, col3) VALUES (1, "ab", "abab");
-- 使用 SELECT 的查询结果插入多行数据
INSERT INTO table_name(col1, col2, col3) SELECT a, b, c FROM table_b;

-- TODO: Update 语句中也可以用子查询
-- 更新数据行: 必须包含 WHERE 子句, 否则会更新所有行
UPDATE table_name SET col1=1, col2='ab', col3='abab' WHERE id=10007;

-- 删除数据行: 必须包含 WHERE 子句, 否则会删除所有行
DELETE FROM table_name WHERE col1 > 0;
```

## 创建/删除表, 约束, 键, 索引, 触发器

最佳实践: 尽量少用外键, 少用约束. 这些限制会降低插入, 删除的效率, 因此最好由应用层来处理这些约束 (即确保插入的数据项满足约束项, 然后进行插入. 或者万一出现不满足约束的情况, 上层应用有兜底机制)

```sql
-- 以下语句构成一个完整的例子

CREATE TABLE User
{
    -- 添加主键约束
    id    CHAR(10)  NOT NULL PRIMARY KEY,
    -- 添加唯一性约束
    username  CHAR(10)  NOT NULL UNIQUE,
    -- 允许/不允许空值
    email CHAR(100) NULL,
    gender CHAR(1)  NOT NULL,
    -- 默认值
    country  CHAR(10) NOT NULL DEFAULT 'ZH',
    -- 检查约束
    accounts_num INTEGER CHECK accounts_num >= 0
};

CREATE TABLE Account
{
    id    CHAR(10)  NOT NULL,
    remain_money INTEGER NOT NULL DEFAULT 0,
    -- 外键约束, 有些 DBMS 支持 cascading delete, 即删除了 user 后, 与之相关联的 account 也会跟着删除
    user_id CHAR(10) NOT NULL REFERENCES User (id)
};

ALTER TABLE User ADD phone CHAR(20) NULL;

-- 并非对所有 DBMS 均有效
ALTER TABLE User DROP COLUMN phone;

-- 创建表后加上约束:
ALTER TABLE Account ADD CONSTRAINT PRIMARY KEY id;
-- ALTER TABLE Account ADD CONSTRAINT FOREIGN KEY (user_id) REFERENCES User (id);
ALTER TABLE User ADD CONSTRAINT CHECK (gender LIKE '[MF]')  -- 只允许取值为 M 或 F, 这里用的是正则表达式


-- 索引
CREATE INDEX name_ind ON User (username);
DROP INDEX name_ind ON User;

-- 触发器 (未必支持, 语法也不尽相同), 触发器是特殊的存储过程
CREATE TRIGGER name_trigger ON User
FOR INSERT, UPDATE
AS
UPDATE User
SET username = Upper(username)
WHERE User.id = inserted.id

-- 因为有外键约束所以先删除 Account
DROP TABLE Account;
DROP TABLE User;
```

## 事务: TRANSACTION, COMMIT, ROLLBACK

transaction (事务) 指的是一组 SQL 语句: COMMIT, ROLLBACK 和 checkpoint 用来保证一个事务要么全部生效, 要么全部不生效


**不同的 DBMS, 语法很不一样**, 以下仅作示意

```sql
BEGIN TRANSACTION
DELETE FROM table_name WHERE col1 > 0;
COMMIT TRANSACTION

BEGIN TRANSACTION
DELETE FROM table_name WHERE col1 > 0;
ROLLBACK

BEGIN TRANSACTION
INSERT INTO User(id, username) VALUES (123, "abc");
SAVE TRANSACTION AddUser;  -- 插入检查点: checkpoint
INSERT INTO User(id, username) VALUES (234, "def");
IF @@ERROR <> 0 ROLLBACK TRANSACTION AddUser;
COMMIT TRANSACTION
```

## 视图: VIEW

## 存储过程

## 游标

## 设计表: 范式
