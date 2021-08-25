size: (width, height)

cv2 在路径中存在中文字符时会无法读写文件

```python
# 写文件
cv2.imencode(".jpg", image)[1].tofile(path)  # image: (H, W, 3) np.ndarray
# 读文件
cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
```

RGB 与 BGR 转换（cv2 默认使用 BGR 格式）

```
# 将srcBGR: uint8数组(H, W, 3)转换为destRGB: uint8数组()
destRGB = cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)
```

