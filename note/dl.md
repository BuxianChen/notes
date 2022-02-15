

## 日志

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
