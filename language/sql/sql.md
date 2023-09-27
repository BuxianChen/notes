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
select datediff('2023-04)
```