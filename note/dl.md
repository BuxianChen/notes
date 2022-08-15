

## 日志

```
nohup python -u main.py > nohup.out 2>&1 &
```

- nohup 表示关闭终端不受影响
- `-u` 表示输出到 `nohup.out` 时不进行缓存操作
- `2>&1` 表示将标准错误与标准输出（即 `print` 函数打印的内容）重定向至 `nohup.out`
- `&` 表示任务在后台进行

对于一个前台运行的程序，可以使用`ctrl+z` 快捷键暂停，或者新开一个终端，执行如下命令让它暂停

```
kill -SIGSTOP <pid>
```

暂停任务再次启动的方法为

```
jobs  # 查看后台任务
fg %1  # 将任务1放到前台继续执行
bg %1  # 将任务1放到后台继续执行
```

或者使用

```
kill -SIGCONT 
```



## 可视化

## 有效利用GPU

## 小工具

```python
import numpy as np
# 输出两位小数, 抑制科学计数法输出
np.set_printoptions(precision=2, suppress=True)
```

捕获 NaN

```
x = torch.tensor([1, 2, np.nan])
torch.isnan(x)
```

## 训练

iteration 1

先设置一个小模型，例如：模型为一层的 LSTM；尝试过拟合一个小数据（注意去除所有例如打乱 dataset 等随机因素），优化器使用 Adam（默认参数）

## 环境配置

高版本的显卡驱动兼容各个低版本的 CUDA 与 CUDNN。

安装多个版本的 CUDA 及 CUDNN：[参考链接](https://towardsdatascience.com/installing-multiple-cuda-cudnn-versions-in-ubuntu-fcb6aa5194e2)
