size: (width, height)

cv2 在路径中存在中文字符时会无法读写文件

```python
# 写文件
cv2.imencode(".jpg", image)[1].tofile(path)  # image: (H, W, 3) np.ndarray
# 读文件
cv2.imdecode(mp.fromfile(path, dtype=np.uint8), -1)
```

