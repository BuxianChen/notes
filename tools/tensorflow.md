# TensorFlow

## tensorflow中卷积操作的padding参数

padding = "same"

$$
out_h = ceil(\frac{in_h}{stride_h})\\
out_w = ceil(\frac{in_w}{stride_w})\\
\text{Note: index of h is 0, index of w is 1}
$$

padding = "valid"

$$
out_h=ceil(\frac{in_h-(filter_h-1)*dilation_h}{stride_h})\\
out_w=ceil(\frac{in_w-(filter_w-1)*dilation_w}{stride_w})
$$

```python
filters = tf.reshape(tf.constant([1., 1., 2.]), (3, 1, 1))
x = tf.reshape(tf.constant([1., 1., 0., 2., 1.]), (1, 5, 1))
tf.nn.convolution(x, filters, strides=2, padding="VALID").numpy().squeeze()
# [2, 4]
tf.nn.convolution(x, filters, strides=2, padding="SAME").numpy().squeeze()
# [3, 5, 3]

filters = tf.reshape(tf.constant([1., 1., 2.]), (3, 1, 1))
x = tf.reshape(tf.constant([1., 1., 0., 2., 1., 2.]), (1, 6, 1))
tf.nn.convolution(x, filters, strides=2, padding="VALID").numpy().squeeze()
# [2, 4]
tf.nn.convolution(x, filters, strides=2, padding="SAME").numpy().squeeze()
# [2, 4, 3]
```

说明：

padding = "valid"时，一定没有填补，并且将结尾多出的部分截断

padding = "same"时，填补方式为对称填补，且结尾优先填补，填补总数为

$$
diation*(k-1)+1+(out-1)*stride-in
$$

附：

$$
floor(\frac{a-1}{b})+1=ceil(\frac{a}{b})
$$

转置卷积：

padding = "same"

padding = "valid"

带output\_padding参数

new\_rows = \(\(rows - 1\)  _strides\[0\] + kernel\_size\[0\] - 2_  padding\[0\] + output\_padding\[0\]\)

new\_cols = \(\(cols - 1\)  _strides\[1\] + kernel\_size\[1\] - 2_  padding\[1\] + output\_padding\[1\]\)

## tf1与tf2的迁移

[https://blog.csdn.net/kyle1314608/article/details/100594884](https://blog.csdn.net/kyle1314608/article/details/100594884)

## tensorflow中dataloader的一个小问题记录

首先，tensorflow里面似乎没有pytorch里DataLoader=Dataset+Sampler的逻辑，tf关于数据处理的基类为：`tf.data.Dataset`。常见的使用方式如下：

```python
import tensorflow as tf
tf_tensor = tf.random.normal((10, 4))
dataset = tf.data.Dataset.from_tensor_slices(tf_tensor)  # staticmethod, 利用tensor构造
# 利用`tf.data.Dataset`对象构造, 返回为`tf.data.Dataset`的继承类
dataset = dataset.shuffle(5).batch(2)
for epoch in range(2):
    for item in dataset:
        print(item.shape)  # 形状为(2, 4)
```

这里的细节在于上述的`dataset`是一个可迭代对象而非一个迭代器，所以每次都会重新打乱样本顺序。

我要做的事情是每次随机丢弃前面若干份数据，再进行shuffle与batch，但希望使用上与上述完全一致，例如：

```python
tf_tensor = tf.random.normal((10, 4))
dataset = MyDataset(tf_tensor, 2)  # 随机丢弃前面0个或1个或2个样本
dataset = dataset.batch(2).shuffle(5)
for epoch in range(2):
    for item in dataset:  # 每次丢弃前面几个样本, 然后batch, 之后以batch为单位打乱顺序
        print(item.shape)  # 形状为(2, 4)
```

```python
class MyDataset(object):
    def __init__(self, tensor, num):
        self.data = tensor
        self.num = num
    def _to_tf_dataset(self):
        begin = np.random.choice(self.num, 1)[0]
        return tf.data.Dataset.from_tensor_slices(self.data[begin:, ...])
    def batch(self, batch_size):
        return self._to_df_dataset().batch(batch_size)
    def shuffle(self, buffer_size):
        return self._to_df_dataset().shuffle(buffer_size)
    def __iter__(self):
        return iter(self._to_tf_dataset())
tf_tensor = tf.random.normal((10, 4))
dataset = MyDataset(tf_tensor, 2)
for epoch in range(2):
    for item in dataset.batch(2).shuffle(5):  
        print(item.shape)
```

```python
class MyDataset():
    def __init__(self, tensor, num):
        self.data = tf.data.Dataset.from_tensor_slices(tensor)
        self.num = num
    def skip(self):
        begin = np.random.choice(self.num, 1)[0]
    def batch(self, batch_size):
        return self.batch(batch_size)
    def shuffle(self, buffer_size):
        return self._to_df_dataset().shuffle(buffer_size)
    def __iter__(self):
        return iter(self._to_tf_dataset())
```

```python
def get_dataset(data, preprocessing):
    """
        returns:
            dataset (tf.data.Dataset): 归一化
    """
```

