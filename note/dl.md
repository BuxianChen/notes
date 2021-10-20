

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

