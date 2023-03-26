
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