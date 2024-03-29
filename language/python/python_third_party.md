# Python Third Party

## numpy

### indexing

indexing 操作指的是 `x[obj]` 这种形式获取参数, 其中 `x` 是 `np.ndarray` 对象, 注意: `x[1, 2]` 与 `x[(1, 2)]` 是完全等价的, 只是语法趟, 按 `obj` 的不同类型, 可以区分为如下几类:

**Basic Indexing**

Basic Indexing 触发的条件如下:

```python
np.newaxis is None  # True
Ellipsis is ...     # True

# Basic Indexing 总是返回一个 view
obj: Union[int, slice, Ellipsis, np.newaxis, List[Union[int, slice, Ellipsis, np.newaxis]]]
```

注意 Basic Indexing 的返回的结果总是原数据的一个 **View**, 这暗示了一个副作用:

```python
x = np.arange(1000000)
y = x[1]
del x   # 不会释放 x, 只是不能使用 x 这个标识符
```

一些例子:

```python
x = np.arange(24)
x.shape = (2, 3, 4)
x[:, np.newaxis, :, :, None].shape  # (2, 1, 3, 4, 1)
x[..., 1:2].shape  # (2, 3, 1), 只能最多出现一个 Ellipsis, 并且 1:2 这种写法会保留这个维度本身
x[..., 1].shape    # (2, 3)

# slice(i, j, k), 首先总是将 i 和 j 转为整数, 然后再区分 k 为正数还是负数, 但区间总是左开右闭: [i, j)
x[0, 0, -1:-3:-1]  # slice(i=-1, j=-3, k=-1), 首先将 i, j 转换为整数, 转换方式为: i = -1 + x.shape[2] = 3, j = -3 + x.shape[2] = 1
# 等价于 x[0, 0, 3:1:-1], 因此取出: [x[0, 0, 3], x[0, 0, 2]] 得到数组 [3, 2]

# 空数组情形
x[1:1, ...].shape   # (0, 3, 4)
```

**Advanced Indexing**【很复杂，待续】

触发条件:

> Advanced indexing is triggered when the selection object, obj, is a non-tuple sequence object, an ndarray (of data type integer or bool), or a tuple with at least one sequence object or ndarray (of data type integer or bool). There are two types of advanced indexing: integer and Boolean.

如果 `obj` 本身是序列类型(但不是元组)或是数组(数据类型可以是bool或int), 或者 `obj` 是一个元组, 但元组至少有一个元素是序列类型或是数组(数据类型可以是bool或int)

> Advanced indexing always returns a copy of the data (contrast with basic slicing that returns a view).

Advanced Indexing 总是返回**复制**

一些例子:

```python
x = np.arange(24)
x.shape = (2, 3, 4)
x[np.array([[0, 1], [1, 0, 0]])].shape  # (2, 3, 3, 4)

y = np.arange(35).reshape(5, 7)
y[np.array([0, 2, 4]), np.array([0, 1, 2])]  # (0, 15, 30)
```


### 奇怪的 id

```python
x = np.array([1, 2])
id(x[0]) == id(x[0])  # 两次取 id 的结果不一样 !!!
```

### topk

```python
idx = np.argpartition(x, k, axis=1) # (m, n) -> (m, n)
x[np.range(x.shape[0]), idx[:, k]]  # (m,) 每行的第k大元素值
```

### save & load

```python
# numpy保存
np.save("xx.npy", arr)
np.load(open("xx.npy"))
```

## pandas

### pandas 的 apply 系列

apply: DataFrame的方法, 可指定axis，应用于行或列

args用于指定额外参数, 但这些参数对于每行或每列是**相同**的

```text
DataFrame.apply(func, axis=0, broadcast=None, raw=False, reduce=None, result_type=None, args=(), **kwds)
```

applymap: DataFrame的方法, 应用于每一个元素

```text

```

| 方法名     | 原型                                                         | 说明               |
| :--------- | :----------------------------------------------------------- | :----------------- |
| `applymap` | `DataFrame.applymap(self, func)`                             | 逐元素操作         |
| `apply`    | `DataFrame.apply(self, func, axis=0, raw=False, result_type=None, args=(), **kwds)` | 按行或列操作       |
| `apply`    | `Series.apply(self, func, convert_dtype=True, args(),**kwds)` | 逐元素操作         |
| `map`      | `Series.map(self, arg, na_action=None)`                      | 替换或者逐元素操作 |


### pandas 分组操作

[官方指南](https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html)

总体上, 从效果上来说分为三个步骤:

- Splitting: 数据分组, 对应的 API 是 `pd.DataFrame.groupby`
- Applying: 一般来说有如下几类, 更一般地, 可以使用 splitting 的结果调用 `apply` 函数
  - Aggregation: 分组后, 对每一组计算一个统计值
  - Transformation: 分组后, 对每一组分别应用于一个变换, 例如对于 A 分组, 将空值填充为 "A", 对 B 分组, 将空值填充为 "B"
  - Filtration: 分组后, 根据一些条件筛选分组内的数据或者筛选整个组地数据, 例如如果一个分组的行数小于 10, 则删除整个分组的数据; 每个分组都只取前 3 行.
- Combining: 将 Applying 的各个分组的结果合并在一次做返回

涉及的 API:
- Splitting: `pd.DataFrame.groupby`: 返回一个 `pandas.api.typing.DataFrameGroupBy` 对象, 此对象有 `groups`, `get_groups` 等基础方法/属性, 也具有下面的 Applying 步骤中的 `agg`/`aggregation`, `transform`, `filter`, `apply` 等方法
- Applying:
  - Aggregation: 内置的方法例如: `mean`, `std`, 更一般地可以使用 `agg`/`aggregation` (这两个方法是一样的, `agg` 只是 short-hand 写法)
  - Transformation: 内置的方法例如: `cumsum`, 更一般地可以使用 `transform`
  - Filtration: 内置的方法例如: `head`, 用于取每组的前几行, 使用自定义函数可以用 `filter`, 但注意 `filter` 只能将整组全部去掉或保留
  - 上面 3 种都不满足时, 可使用 `apply` 函数

**Splitting: groupby 与 pandas.api.typing.DataFrameGroupBy 对象的方法**

```python
df = pd.DataFrame({
    "A": ["a", "a", "b", "b"],
    "B": [1, 9, 2, 4],
    "C": [4, 6, 1, 10],
})
grouped = df.groupby('A')

for name, group in grouped:
    print(name)   # "a"/"b"
    print(group)  # (2, 3) shape DataFrame

grouped.get_group("a")  # (2, 3) shape DataFrame
grouped.groups          # {'a': [0, 1], 'b': [2, 3]}
```

**Applying: `agg`、`transform`, `filter`, `apply`**

- `agg`、`transform`, `filter`, `apply` 传参时的自定义函数 (UDF: User-Defined Function) 的输入输出条件 (官方文档似乎对此语焉不详) 不尽相同
- 这几个方法最终 Combining 之后的 DataFrame 的 index 的形式有所不同, 这里只讨论 `group(..., group_keys=True)` 的情况:
    - `apply` 在 UDF 的出参是一个 DataFrame 的情况下, 会把 groupby 的列作为索引并保留原始索引以构成两级索引
    - `transform` 会把 groupby 的列丢弃, 原本的索引依然作为索引
    - `agg` 会把 groupby 的列作为索引, 原本索引丢弃
    - `filter` 除去索引可能会变少外, groupby 列被保留为列

太长不看系列 (`transform`, `agg`, `filter` 都可用 `apply` 实现):

```python
# transform
fn = lambda x: x+1 if x.name=='a' else x-100
df.groupby('State', group_keys=True).transform(fn)      # Input: Series, Output: Series(List) (Same length with Input)
apply_fn = lambda x: pd.DataFrame([fn(x[c]) for c in x.columns]).T
df.groupby('State', group_keys=False).apply(apply_fn)   # Input: DataFrame, Output: DataFrame

# agg
fn = lambda x: x.sum() if x.name=='a' else 0
df.groupby('State', group_keys=True).agg(fn)            # Input: Series, Output: Scalar
apply_fn = lambda x: pd.Series({c: fn(x[c]) for c in x.columns if c not in ["State"]})
df.groupby('State', group_keys=False).apply(apply_fn)   # Input: DataFrame, Output: Series

# filter
fn = lambda x: True
df.groupby('State', group_keys=True).filter(fn)         # Input: DataFrame, Output: bool
apply_fn = lambda x: x if fn(x) else []
df.groupby('State', group_keys=False).apply(apply_fn)   # Input: DataFrame, Output: DataFrame

# apply
df.groupby('State', group_keys=True).apply(lambda x: x+1)    # Input: Dataframe, Output: Scalar/Series(List)/DataFrame

# 当UDF Output是标量时, apply 的最终结果是 Series
df.groupby('State', group_keys=False).apply(lambda x: 1)     # Final Output: Series
```

参考资料:

- StackOverflow 问答: [https://stackoverflow.com/questions/27517425/apply-vs-transform-on-a-group-object](https://stackoverflow.com/questions/27517425/apply-vs-transform-on-a-group-object)
- 官方指南: [https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html](https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html)

UDF 的输入输出对比

**transform**

输入: 每组的每一列(Series)作为输入
输出: 与输入同等长度的列表/Series

**agg**

输入: 每组的每一列(Series)作为输入
输出: 标量

**filter**
输入: 每组的DataFrame作为输入
输出: True/False

**apply**
输入: 每组的DataFrame作为输入
输出: 标量/Series(不必与输入同等长度)/DataFrame


为了弄清这些函数的 UDF 的输入输出, 可以构造类似如下的测试代码

```python
import pandas as pd
import numpy as np
from IPython.display import display
df = pd.DataFrame({'State':['Texas', 'Texas', 'Florida', 'Florida'], 
                   'a':[4,5,1,3], 'b':[6,10,3,11]})
def subtract_two(x):
    display(x)
    print()
    y = x['a'] - x['b']
    display(y)
    print()
    return y

result = df.groupby('State').apply(subtract_two)
display(result)
print()
print(result.to_numpy())
print(result.index)
```

输出结果如下:
```
     State  a   b
2  Florida  1   3
3  Florida  3  11

2   -2
3   -8
dtype: int64

   State  a   b
0  Texas  4   6
1  Texas  5  10

0   -2
1   -5
dtype: int64

State     
Florida  2   -2
         3   -8
Texas    0   -2
         1   -5
dtype: int64

[-2 -8 -2 -5]
MultiIndex([('Florida', 2),
            ('Florida', 3),
            (  'Texas', 0),
            (  'Texas', 1)],
           names=['State', None])
```

pandas 可能会自动做些优化 (与numba有关), 所以有些像上面那种测试代码可能会有比较诡异的结果, 例如:

```python
df = pd.DataFrame({'State':['Texas', 'Texas', 'Florida', 'Florida'], 
                   'a':[4,5,1,3], 'b':[6,10,3,11]})
def subtract_two(x):
    if x.name == 'a':
        y = x + 1
    else:
        y = x + 2
    display(type(x), type(y), "\n", x, "\n", y)
    print()
    return y

# def inspect(x):
#     print(type(x))
#     raise

result = df.groupby('State').transform(subtract_two)
display(result)
print()
print(result.to_numpy())
print(result.index)
```

### pandas读写excel文件

[参考链接1](https://pythonbasics.org/read-excel/), [参考链接2](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.ExcelWriter.html)

pandas读写excel依赖xlrd, xlwt包, \(ps: 可以尝试直接使用这两个包直接进行读写excel文件\)

```python
df1 = pd.DataFrame({"A": [1, 2, 3]})
df2 = pd.DataFrame({"B": [2, 0, 3]})
df3 = pd.DataFrame({"C": [3, 2, 3]})
with pd.ExcelWriter("path_to_file.xlsx", engine="openpyxl") as writer:
    df1.to_excel(writer, sheet_name='Sheet1')
    df2.to_excel(writer, sheet_name="页2")
with pd.ExcelWriter("path_to_file.xlsx", engine="openpyxl", mode="a") as writer:
    df3.to_excel(writer, sheet_name="Sheet3", index=False)
test = pd.read_excel("path_to_file.xlsx", sheet_name=[0, "Sheet3"])
print(type(test))  # <class 'dict'>
print(test.keys())  # dict_keys([0, 'Sheet3'])
```

直接使用xlrd包示例: [参考链接](https://www.codespeedy.com/reading-an-excel-sheet-using-xlrd-module-in-python/)

```python
# 直接使用xlrd包
import xlrd
wb = xlrd.open_workbook("path_to_file.xlsx")
sheet = wb.sheet_by_index(0)
sheet = wb.sheet_by_name("Sheet3")
# <class 'xlrd.book.Book'> <class 'xlrd.sheet.Sheet'> 4 1
print(type(wb), type(sheet), sheet.nrows, sheet.ncols)
for i in range(sheet.nrows):
    for j in range(sheet.ncols):
        print(sheet.cell_value(i, j), end=" ")
    print()
```

直接使用xlwt包示例: [参考链接](https://www.codespeedy.com/reading-an-excel-sheet-using-xlrd-module-in-python/)

```python
# Writing to an excel sheet using Python 3.x. or earlier 
import xlwt as xw

# Workbook is created 
wb = xw.Workbook() 

# add_sheet is used to create sheet. 
sheet1 = wb.add_sheet('Sheet 1') 
# Specifying style of the elements 
style_value1= xw.easyxf('font: bold 1')
style_value2 = xw.easyxf('font: bold 1, color blue;')
# Input data into rows 
sheet1.write(1, 0, 'Code Speedy', style_value1) 
sheet1.write(2, 0, 'Sarque Ahamed Mollick', style_value2) 

# Input data into columns
sheet1.write(0, 1, 'Position') 
sheet1.write(0, 2, 'No of Posts') 

# 似乎不能写为以.xlsx为后缀的文件(运行不报错, 但使用Excel2019打不开)
wb.save('xlwt codespeedy.xls')  # .xls文件能用Excel2019打开
```

某些情况下`.xlsx`被自动转为了`.xlsm`格式, 可以用pandas进行修复, 注意下面的例子也演示了如何获取一个excel文档的所有sheet名称

```python
x = pd.ExcelFile(r"C:\Users\chenbx\Desktop\调优\默认值.xlsm")
sheet_names = x.sheet_names
y = pd.ExcelWriter(r"C:\Users\chenbx\Desktop\调优\默认值.xlsx")
for sheet in sheet_names:
    df = pd.read_excel(r"C:\Users\chenbx\Desktop\调优\默认值.xlsm", sheet_name=sheet)
    df.to_excel(y, sheet_name=sheet)
y.save()
```

### pandas index相关的操作

```python
# DataFrame.set_index(keys, drop=True, append=False, inplace=False, verify_integrity=False)
df.set_index("key", drop=True)  # 将df["key"]这一列作为新的index, 将原有的index丢弃
df.reset_index(drop=True)  # 将原有的index丢弃, 新的index为默认的[0,1,...], 丢弃的index不作为新列
df.reindex(list_or_index, fill_value=0)  # 只保留list_or_index中的行, 用0填补不存在的行
df.rename(index={1: -1}, columns={"a": "b"}, inplace=False) # 对行或列重命名
```

### merge, join技巧

```python
# 需求: df1与df2, 需要按指定的列名col1, col2做内连接, 希望输出两个dataframe:
# new_df1: 能连上的df1中的部分, 行的相对顺序与df1保持一致, 且列名与df1完全一致
# new_df2: 能连上的df2中的部分, 列名与df2完全一致

# 注:
# 1) 为什么不能用普通的表连接: pandas的dataframe不允许两列的列名相同(实际需求中, df1的列与df2中的列名可能有重复, 并且这些列代表着完全不同的含义)
# 2) col1与col2可以相同, 也可以不同

# 在df1和df2的index都不重要时, 可以使用如下方法
def mymerge(df1, df2, col1, col2):
    df1_new = pd.merge(df1, df2[[col2]].set_index(col2, drop=True), left_on=col1, right_on=col2, how="inner")
    df2_new = pd.merge(df2, df1[[col1]].set_index(col1, drop=True), left_on=col2, right_on=col1, how="inner")
    return df1_new, df2_new
```

## easydict/addict/dotmap

这几个包均是对 python 字典这一基本数据类型的封装，使得字典的属性可以使用点来访问，具体用法及区别待补充：

```python
a.b # a["b"]
```

一些开源项目对这些包的使用情况：

- addict：mmcv
- easydict：
- dotmap：[MaskTheFace](https://github.com/aqeelanwar/MaskTheFace/blob/master/utils/read_cfg.py)


## 语言检测模块

`langdetect`, `langid`等

## 发送邮件模块

[未仔细校对过](https://blog.csdn.net/tianshishangxin1/article/details/109856352)

## 压缩与解压模块

### zipfile模块

参考链接: [https://www.datacamp.com/community/tutorials/zip-file\#EZWP](https://www.datacamp.com/community/tutorials/zip-file#EZWP)

```python
# 解压文件
import zipfile
zipname = r'D:\work0126\aa.zip'
out_dir = r"C:\work0126"
pswd = '12345'
with zipfile.ZipFile(zipname) as file:
    # password you pass must be in the bytes you converted 'str' into 'bytes'
    file.extractall(path=out_dir, pwd = bytes(pswd, 'utf-8'))
# 打包为zip
```

## pyhanlp

### 安装说明(1.7.8版本)

**step 1**

首先安装JPype1==0.7.0(版本号必须完全一致)

```text
pip install JPype1-0.7.0-cp37-cp37m-win_amd64.whl
```

**step 2**

接下来安装pyhanlp\(直接去[网站](https://github.com/hankcs/pyhanlp)下载代码[pyhanlp-master.zip](https://github.com/hankcs/pyhanlp/archive/master.zip), 注意项目名为pyhanlp\)

并下载: jar与配置文件[hanlp-1.7.8-release.zip](http://nlp.hankcs.com/download.php?file=jar), 数据文件[data-for-1.7.5.zip](http://nlp.hankcs.com/download.php?file=data)

注意data1.7.5是被1.7.8版本hanlp兼容的\(实际上也没有data1.7.5版本\), 至此原料已经准备齐全

首先将pyhanlp-master.zip解压, 并进入该目录用如下方式安装

```text
python setup.py install
```

接下来进入安装位置例如:

`C:\Users\54120\anaconda3\envs\hanlp_copy\Lib\site-packages\pyhanlp-0.1.66-py3.7.egg\pyhanlp\static`

将`data-for-1.7.5.zip`解压后的`data`文件夹, `hanlp-1.7.8-release.zip`解压后的`hanlp-1.7.8-sources.jar`, `hanlp-1.7.8.jar`, `hanlp.properties`都放入上述目录下, 最终此目录的结构为:

```text
static
|-  data
    |-  dictionary
    |-  model
    |-  test  (后续示例代码可能将数据下载到这个目录)
    |-  README.url
    |-  version.txt  (内容为1.7.5)
|-  hanlp-1.7.8-sources.jar
│-  hanlp-1.7.8.jar
│-  hanlp.properties
│-  hanlp.properties.in
│-  index.html
│-  README.url
│-  __init__.py
```

**step 3**

修改`hanlp.properties`文件的内容

```text
root=C:/Users/54120/anaconda3/envs/hanlp_copy/Lib/site-packages/pyhanlp-0.1.66-py3.7.egg/pyhanlp/static
```

**step 4**

检查, 在命令行输入

```text
hanlp -v
jar  1.7.8-sources: C:\Users\54120\anaconda3\envs\hanlp_copy\lib\site-packages\pyhanlp-0.1.66-py3.7.egg\pyhanlp\static\hanlp-1.7.8-sources.jar
data 1.7.5: C:\Users\54120\anaconda3\envs\hanlp_copy\Lib\site-packages\pyhanlp-0.1.66-py3.7.egg\pyhanlp\static\data
config    : C:\Users\54120\anaconda3\envs\hanlp_copy\lib\site-packages\pyhanlp-0.1.66-py3.7.egg\pyhanlp\static\hanlp.properties
```

另外, python中应该也要确保可以正常导入

```text
python -c "import pyhanlp"
```

**注意**

上述繁琐的过程使得环境迁移时除了拷贝envs还要修改配置文件.

## spacy

### 下载模型

解决类似如下命令因为网络原因失效的方法:

```text
python -m spacy download en_core_web_sm
```

去[https://github.com/explosion/spacy-models/查看相应的版本号](https://github.com/explosion/spacy-models/查看相应的版本号), 下载类似如下链接的文件

```text
https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.0.0/de_core_news_sm-3.0.0.tar.gz
```

```text
pip install xxx.tar.gz
```

更为详细的解释如下（spacy2.1.9版本源码分析）

执行 `python -m spacy download en_core_web_sm` 实际调用 `site-packages/spacy/__main__.py`。之后调用了

```python
# res = requests.get("https://raw.githubusercontent.com/explosion/spacy-models/master/shortcuts-v2.json")
# res.json()

# 确定下载的模型版本
res = requests.get("https://raw.githubusercontent.com/explosion/spacy-models/master/compatibility.json")
version = res.json()["spacy"]["2.1.9"]['en_core_web_sm']  # 2.1.0为2.1.9版本相匹配的en_core_web_sm模型

# 最后实际调用了
# python -m pip install --no-cache-dir --no-deps <download-url>

download_url = "https://github.com/explosion/spacy-models/releases/download/"
+ "en_core_web_sm-2.1.0/en_core_web_sm-2.1.0.tar.gz#egg=en_core_web_sm==2.1.0"

m = "en_core_web_sm"
v = "2.1.0"
format = "{m}-{v}/{m}-{v}.tar.gz#egg={m}=={v}"
```

## huggingface transformers

### 基本使用

### 模型下载目录

设置模型下载位置可参见[官网介绍](https://huggingface.co/transformers/installation.html), 摘抄如下:

**Caching models**

This library provides pretrained models that will be downloaded and cached locally. Unless you specify a location with `cache_dir=...` when you use methods like `from_pretrained`, these models will automatically be downloaded in the folder given by the shell environment variable `TRANSFORMERS_CACHE`. The default value for it will be the Hugging Face cache home followed by `/transformers/`. This is \(by order of priority\):

* shell environment variable `HF_HOME`
* shell environment variable `XDG_CACHE_HOME` + `/huggingface/`
* default: `~/.cache/huggingface/`

So if you don’t have any specific environment variable set, the cache directory will be at `~/.cache/huggingface/transformers/`.

**Note:** If you have set a shell environment variable for one of the predecessors of this library \(`PYTORCH_TRANSFORMERS_CACHE` or `PYTORCH_PRETRAINED_BERT_CACHE`\), those will be used if there is no shell environment variable for `TRANSFORMERS_CACHE`.

### 开源模型

发现有英翻中的模型, [开源模型目录](https://huggingface.co/models), 搜索`zh`, \([https://huggingface.co/Helsinki-NLP/opus-mt-en-zh](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh)\)

使用方法:

```python
# 前三行参照模型地址
# https://huggingface.co/Helsinki-NLP/opus-mt-en-zh/tree/main
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")

# 后面三行参照transformers文档
# https://huggingface.co/transformers/task_summary.html#translation
inputs = tokenizer.encode("translate English to German: Hugging Face is a technology company based in New York and Paris", return_tensors="pt")
outputs = model.generate(inputs, max_length=40, num_beams=4, early_stopping=True)
print(tokenizer.decode(outputs[0]))
```

### 离线模型下载实例

`EncoderClassifier` 中有如下注释：

```
classifier = EncoderClassifier.from_hparams(
    ...     source="speechbrain/spkrec-ecapa-voxceleb",
    ...     savedir=tmpdir,
    ... )
```

#### 离线下载模型步骤如下：

```
# 需要先安装git-lfs
git clone https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
# 将hyperparams.yaml中的pretrained_path修改为/home/buxian/Desktop/spkrec-ecapa-voxceleb
```

这样便可以直接使用如下方式导入模型（完全绕过默认路径 `~/.cache/huggingface/hub`）

```
from speechbrain.pretrained import EncoderClassifier
classifier = EncoderClassifier.from_hparams(source="/home/buxian/Desktop/spkrec-ecapa-voxceleb")
```

备注：此处的 git clone 这一方法在离线下载时具有通用性，而修改 `pretrain_path` 是 `speechbrain` 包的内部的逻辑造成的。如果不修改 `pretrain_path`，将无法绕过默认下载路径 `~/.cache/huggingface/hub`。

## 读写excel(xlsxwriter与pandas)

pandas与xlsxwriter均支持给输出的excel自定义格式.

```python
# 注意workbook指的是一个excel文件, 而worksheet指的是excel文件当中的一个sheet
import xlsxwriter
workbook  = xlsxwriter.Workbook('filename.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write(0, 0, 'Hello Excel')
workbook.close()
```

```python
# 在生成的excel中操作: “条件格式->管理规则”就可以看到这里定义的规则
import pandas as pd
df = pd.DataFrame({'Data': [1, 1, 2, 2, 3, 4, 4, 5, 5, 6]})
writer = pd.ExcelWriter('conditional.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1', index=False)
workbook  = writer.book
worksheet = writer.sheets['Sheet1']
format1 = workbook.add_format({'bg_color': '#FFC7CE', # 粉色
                               'font_color': '#9C0006'})  # 深红色
format2 = workbook.add_format({'bg_color': '#C6EFCE', # 青色
                               'font_color': '#006100'})  # 深绿色
worksheet.conditional_format('A1:A8', {'type': 'formula', 'criteria': '=MOD(ROW(),2)=0', 'format': format1})
worksheet.conditional_format('A1:A8', {'type': 'formula', 'criteria': '=MOD(ROW(),2)=1', 'format': format2})
writer.save()
```

**对单元格的一些字符变成红色字符, 另一些字符仍然为黑色**

```python
import xlsxwriter
workbook = xlsxwriter.Workbook('x.xlsx')
worksheet = workbook.add_worksheet()
cell_format_red = workbook.add_format()
cell_format_red.set_font_color('red')
cell_format_black = workbook.add_format()
cell_format_black.set_font_color('black')

worksheet.write_rich_string(0, 0, cell_format_red, "我", "不", cell_format_black, "是", cell_format_red, "谁")
workbook.close()
```

**数据验证(单元格内的值必须满足一定的条件)**

```python
import xlsxwriter
workbook = xlsxwriter.Workbook('x.xlsx')
worksheet = workbook.add_worksheet()
validation = {"validate": "list", "value": ["apple", "banana", "orange"]}
# 对一个区域内的单元格设置条件, 注意start与end也包括在内
worksheet.data_validation(start_row, start_col, end_row, end_col, validation)
```

## html转pdf的(pdfkit)

依赖于[wkhtmltopdf](https://wkhtmltopdf.org/downloads.html), 安装后\(windows上需添加至环境变量\)可以利用pdfkit包进行html到pdf的转换, 实际体验感觉对公式显示的支持不太好.

```python
# pip install pdfkit
import pdfkit
pdfkit.from_url('https://www.jianshu.com','out.pdf')
```

## black（自动将代码规范化）

black模块可以自动将代码规范化\(基本按照PEP8规范\), 是一个常用工具

```text
pip install black
black dirty_code.py
```

## albumentations（待补充）

基于opencv的数据增强包

## natsort

```python
from natsort import natsorted
x = ["1.png", "10.png", "2.png"]
sorted_x = natsorted(x)
# sorted_x: ["1.png", "2.png", "10.png"]
```

## yacs

作者为 faster rcnn 的作者 Ross Girshick，用于解析 yaml 文件

## timeout_decorator

超时自动退出装饰器

## redis

python调用redis服务主要是如下两个第三方包:

- 包名(pip install): redis, import时的模块名: redis
- 包名(pip install): redis-py-cluster, import时的模块名: rediscluster

以上两个包存在一些版本兼容性问题: 

- redis-py-cluster依赖于redis, 但目前redis已经集成了redis-py-cluster的所有内容, 使用方式为:
    ```python
    from rediscluster import RedisCluster  # 旧版, redis-py-cluster 2.x, 依赖于redis 3.x
    from redis.cluster import RedisCluster  # redis 4.x
    ```
    [redis-py-cluster github README](https://github.com/Grokzen/redis-py-cluster)
    > In the upstream package redis-py that this librar extends, they have since version * 4.1.0 (Dec 26, 2021) ported in this code base into the main branch.
    
    [redis-py readthdocs](https://redis-py.readthedocs.io/en/latest/clustering.html)
    > The cluster client is based on Grokzen’s redis-py-cluster, has added bug fixes, and now supersedes that library. Support for these changes is thanks to his contributions
- redis包在旧版本中存在`Redis`与`StrictRedis`两个类, 但在目前的 `3.x` 及以上版本已经合并为一个类, 源码中有如下代码:
    ```
    StrictRedis = Redis
    ```
    关于StrictRedis与Redis的讨论参考[stackoverflow](https://stackoverflow.com/questions/19021765/redis-py-whats-the-difference-between-strictredis-and-redis)

- 结论: 不要使用`redis-py-cluster`, 直接安装`redis 4.x`及以上版本, 使用`redis.Redis`和`redis.cluster.RedisCluster`类, 不要使用`redis.StrictRedis`


具体使用方式为:

```python
from redis import Redis
from redis.cluster import ClusterNode, RedisCluster

# decode_respose为True表示利用get得到的数据类型为字符串
redis_service = Redis(
    host="127.0.0.1",
    port=6379,  # redis默认端口为6379
    decode_responses=True,
    password="xxx"
)

redis_service = RedisCluster(
    startup_nodes=[
        ClusterNode("127.0.0.1", 6379)
    ],
    decode_responses=True,
    password="xxx"
)

# 以下操作适用于Redis与RedisCluster
key = "test001"
redis_service.exists(key)
redis_service.set(key, json.dumps(["text1", "text2"]))
redis_service.set(key, json.dumps(["text1", "text2", "text3"]))
value = redis_service.get(key)
redis_service.delete(key)
```

## pytest

经过阅读多篇相关的博客, 总结如下, python 包的项目组织形式严格按照如下方式进行

备注: 很多项目其实未必采用了“正确”的方式组织代码，“正确”的含义也会随着时间的推移而改变
```
- src
  - package_name
    - submodule1/  # 全部加上__init__.py
      - module_a/
        - __init__.py
        - some.py
      somename.py
      - __init__.py
- tests/
  - __init__.py
  - test_a/
    - test_1.py
    - __init__.py
  - test_b/
    - test_2.py
    - __init__.py
  test_c.py
setup.py  # 尽量不写setup.py, 完全由pyproject.toml配置
setup.cfg
pyproject.toml
```

**关于安装**

关于 `pyproject.toml` 与 `setup.py` 与 `setup.cfg`：这三个文件与 `pip install -e .` 或 `pip install` 的行为相关。

- `pyproject.toml`：可以配置打包工具，也可以配置包的一些基本信息。
  - 当打包工具项配置为 `setuptools.build_meta` 时，那么 `pip` 会去按照 `setup.py` 的逻辑去执行安装命令，`pyproject.toml` 和 `setup.cfg` 的其余配置项也会被自动用于 `setup.py` 的执行逻辑里
  - 当打包工具配置为其他选项例如：`hatchling.build` 时，那么 `pip` 会忽略 `setup.py`，而按照 `pyproject.toml` 和 `setup.cfg` 的其余配置项执行安装命令。
- `setup.py` 与 `pyproject.toml` 同时存在时，执行 `pip install .` 命令时会按照 `pyproject.toml` 来执行。当然，执行 `python setup.py install` 安装时则忽略 `pyproject.toml`。但一般情况下，推荐使用 `pip install .` 而非 `python setup.py install`。

**关于测试**

关于 `__import__`、`import`、`importlib.import_module`：[stackoverflow问答](https://stackoverflow.com/questions/28231738/import-vs-import-vs-importlib-import-module)

`import` 实际上最终是调用了 `__import__`，而 `__import__` 在底层是直接使用了 C 代码来实现。`importlib.import_module` 本质上的行为跟 `__import__` 比较类似，只是实现方式上是使用纯 Python 来实现的。具体解释如下：

```
tests
  - __init__.py
  - test_a/
    - __init__.py
    - test_b.py
```

- `import tests.test_a.test_b`：`sys.modules` 增加 `tests`、`tests.test_a`、`tests.test_a.test_b` 这几个模块，当前文件可以使用 `tests.test_a.test_b` 这个命名空间，即可以使用 `tests.test_a.test_b.xx`。
- `mod = importlib.import_module("tests.test_a.test_b")`：`sys.modules` 增加 `tests`、`tests.test_a`、`tests.test_a.b` 这几个模块，当前文件可以使用 `mod` 这个命名空间，即可以使用 `mod.xx`。
- 在进行了上面两种方式之一进行导入后，可以直接使用 `sys.modules` 获取 `tests.test_a`：
    ```
    test_a_mod = sys.modules['tests.test_a']
    print(test_a_mod.xxx)
    ```

pytest 在处理 import 的问题时, 支持了三种方式 `prepend`, `append` 与 `importlib`。但每种方式都有各自的缺点。目前的最佳实践是：

- **包放在 `src` 目录下**，所有的模块各个目录应显式地添加 `__init__.py`。放入 `src` 目录最主要的目的是使得本地开发环境与 release 到 PyPI 后别人使用 `pip install` 的方式安装的环境相同。
- **测试代码完全按包的形式组织**，即各个目录显式地添加 `__init__.py`，独立于包之外，即与 `src` 目录同级方式 `tests` 文件夹。
- **测试的目的是包在安装之后的行为是否正常**，测试前应该以 `pip install -e .` 或 `pip install .` 的方式将包安装。这也是包要放在 `src` 目录下的原因，即保证不会意外导入当前路径下的代码（很多情况下，`sys.path` 变量会把当前路径添加至模块搜索路径，这样子可能会意外导入）。
- **以默认的 `prepend` 作为 pytest 的导入方式**。注意：按照 pytest 内部的逻辑，使用 `prepend` 作为导入方式，不可避免地会修改 `sys.path`，但测试代码完全按包的形式组织，已经可以尽可能小的避免了 `sys.path` 的修改，但好处是 `tests` 目录下的各个文件之间可以相互 import。import 的方式应为 `import tests.xx.yy`。然而使用 `importlib` 作为导入方式，测试文件之间无法进行相互 import，这是一个重要的缺点。
- **执行 `pytest` 命令**（即不要以 `python -m pytest`的方式启动）：使用 `python -m pytest` 会将当前目录添加至 `sys.path` 目录，因此要避免使用。


## 代码片段

```python
def int2str(x):
    return tuple(str(i) if isinstance(i, int) else int2str(i) for i in x)
x = ((1, 2), 1, 3)
int2str(x)  # 输出(('1', '2'), '1', '3')

# 一个综合的例子
from functools import wraps
def to_string(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        def int2str(x):
            return tuple([str(i) if isinstance(i, int) else int2str(i) for i in x])
        return int2str(func(*args, **kwargs))
    return wrapper
@to_string
def f():
    """asd"""
    return ((1, 2), 1, 3)
f.__doc__
```

```python
# for else字句, 若正常结束, 则执行else语句
for n in range(2, 8):
    for x in range(2, n):
        if n % x == 0:
            print( n, 'equals', x, '*', n/x)
            break
    else:
        # loop fell through without finding a factor
        print(n, 'is a prime number')
```

```python
# 慎用, 速度很慢!!
df = pd.DataFrame({"A": [1, 2, 3], "B": [0, 1, 2]})
df[["E", "F"]] = df["A"].apply(lambda x: pd.Series((x, x)))
# 快的方式待整理
```

```python
# 用两个列表创建字典的较快方式(似乎快于字典推导式)
x = dict(zip(key, value))
```

**打印带颜色的文本**

```python
# [0;31m 红色 [0;32m 绿色 [0;33m 黄色
s = "红色"
s = "\033[0;31m" + s + "\033[0m"
```