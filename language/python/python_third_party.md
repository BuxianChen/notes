# Python Third Party

## numpy

```python
idx = np.argpartition(x, k, axis=1) # (m, n) -> (m, n)
x[np.range(x.shape[0]), idx[:, k]]  # (m,) 每行的第k大元素值
```

```python
# numpy保存
np.save("xx.npy", arr)
np.load(open("xx.npy"))
```

## pandas

#### pandas的apply系列

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