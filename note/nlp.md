# 基于字/词的 NLP 技术

## TF-IDF & BM25

### TF-IDF

TF-IDF 用于计算文档中的每个单词的重要性，这样便可以得到文档中的关键词，问题定义为：给定一个文本集合 $$D=\{d_i\}_{i=1}^{N}$$，对于每个文档 $$d$$，需要计算文档 $$d$$ 中每个单词 $$w$$ 的重要性

计算公式为：

$$
tf-idf(w, d) = tf(w, d)\cdot idf(w, d)
$$

**term frequecy(词频): $$tf(w, d)$$**

$$
tf(w,d) = \frac{f(w,d)}{\sum_{d\in D}{f(w, d)}}
$$

其中 $$f(w,d)$$ 为单词 $$w$$ 在文档 $$d$$ 中出现的次数。

**inverse document frequency(逆文档词频): $$idf(w, d)$$**

$$
idf(w, d)=\log(\frac{N}{df_{w}+1})
$$

其中 $$df_{w}$$ 在文档底库中有多少个文档包含词 $$w$$。


### BM25

BM 算法用于文本检索 (elasticsearch 2.x 版本所使用的算法)。检索问题的定义如下：给定一个文本集合 $$D=\{d_i\}_{i=1}^{N}$$，输入一个文本 $$q$$，按相关性得到底库 $$D$$ 中最相关的文本。

BM25 算法框架为：对于每个底库文档 $$d$$，计算查询 $$q$$ 与文档 $$d$$ 的相似度，之后按相似度排序即可。

具体计算相似度的方法为：

$$
Score(q, d) = \sum_{i=1}^{n}W_iR(q_i, d)
$$

其中 $$n$$ 表示查询 $$q$$ 中不同的单词的个数。

**单词权重：$$W_i$$**

其中 $$q_i$$ 为 $$q$$ 的每个单词（**如果某个单词出现了多次，求和项中只计算一次**），$$W_i$$ 为每个单词的权重。而 $$W_i$$ 的定义为：

$$
W_i = IDF(q_i) = \log(\frac{N-df_{q_i}+0.5}{df_{q_i}+0.5}+1)\in(0, \log(2N+1))
$$

其中 $$df_{q_i}$$ 为底库中包含单词 $$q_i$$ 的文档个数。因此词 $$q_i$$ 在每个文档出现的越频繁（例如停用词），则其对搜索结果的权重越小。


**单词相关度：$$R(q_i, d)$$**

而 $$R(q_i, d)$$ 的定义为：

$$
R(q_i, d) = \frac{(k_1+1)tf(q_i, d)}{tf(q_i, d)+k_1(1-b+b\frac{L_d}{L_{ave}})}
$$

其中 $$k_1$$ 与 $$b$$ 为超参数，一般设置为 $$k_1\in[1.2, 2.0], b=0.75$$, $$L_d$$ 为文档 $$d$$ 的长度, $$L_{ave}$$ 为文档底库 $$D$$ 的平均长度。而 $$tf(q_i,d)$$ 即为 *term frequency*：

$$
tf(q_i,d) = \frac{f(q_i,d)}{\sum_{d\in D}{f(q_i, d)}}
$$

其中 $$f(q_i,d)$$ 为单词 $$q_i$$ 在文档 $$d$$ 中出现的次数。

**修正项（非必须，且不同 BM25 算法对此项的处理略有不同）**

当 $$q$$ 比较长时，还需要对权重 $$W_i$$ 做一步修正：

$$
S(q_i, q) = \frac{(k_3+1)tf(q_i, q)}{k_3 + tf(q_i, q)}
$$

**完整公式**

$$
Score(q, d) = \sum_{i=1}^{n}{\frac{(k_3+1)tf(q_i, q)}{k_3 + tf(q_i, q)}\cdot\log(\frac{N-df_{q_i}+0.5}{df_{q_i}+0.5}+1)\cdot\frac{(k_1+1)tf(q_i, d)}{tf(q_i, d)+k_1(1-b+b\frac{L_d}{L_{ave}})}}
$$

其中 $$tf(w, d)$$ 表示 *term frequency* (单词 $$w$$ 在文档 $$d$$ 中出现的次数)，而 $$df_{w}$$ 在文档底库中有多少个文档包含词 $$w$$。

# 评估指标

## BLEU

# 预训练模型

## Tokenizer

### BPE 算法

**动机**

对于英文来说，词的数量可能太多。同一个词根，可以通过修改前/后缀的方式来得到动词形式，名词形式，副词形式，例如如下词表：

own、owner、play、player、research、researcher、care、careful、hope、hopeful。

使用词作为 embeding 的最小单位，需要 10 个向量。但如果拆解为：

own、play、research、care、hope、er、ful。

那么仅需要 7 个向量，且这些词根和前/后缀有着某种含义。因此 embeding 的对象为这些 word piece，似乎比较合理。这带来一个问题，上面的拆解过程是人工找的规律（需要语言学的专业知识），因此需要用一个算法来自动发现这些 word piece。BPE 算法就是一种自动发现 word piece 的算法

具体算法流程参考博客：[Byte Pair Encoding](https://towardsdatascience.com/byte-pair-encoding-subword-based-tokenization-algorithm-77828a70bee0)

# NLP 小工具（常用正则、文本预处理工具等）

## 英文正则化(上撇号缩写等)

```python
import contractions
contractions.fix("I'm")  # "I am"
```

# Draft

## Causal Language Modeling (CausalLM, LM)

即普通的语言模型: 根据已经见到的内容预测下一个 token

参考博客: https://www.projectpro.io/recipes/what-is-causal-language-modeling-transformers

The task of predicting the token after a sequence of tokens is known as causal language modeling. In this case, the model is just concerned with the left context (tokens on the left of the mask).

