# OpenCV

## opencv-python

### 图像格式

cv2 包里函数参数 size 一般为 (width, height)，但图像的表示形式上，使用 numpy 数组来表示，例如：`arr` 为 3 维 dtype = np.uint8，shape = (height, width, 3)，颜色通道依次为 BGR 。

cv2 对 numpy 数组进行操作时往往需要 numpy 数组的内部存储是 C 连续的，numpy 中对数组的某些操作会使得数组变的不是 C 连续的，例如：

- np.transpose

```python
arr = np.ascontiguousarray(arr)
```

### cv2 图像读写操作

cv2 在路径中存在中文字符时会无法读写文件

```python
# 写文件
cv2.imencode(".jpg", image)[1].tofile(path)  # image: (H, W, 3) np.ndarray, uint8, BGR format
# 读文件
image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)  # image: (H, W, 3) np.ndarray, uint8, BGR format
```

备注：imdecode 中的第二个参数 -1 实际上是 cv2.IMREAD_UNCHANGED。对于 imread 函数来说，原型如下：

```python
cv2.imread(filename, flags=cv2.IMREAD_COLOR)
```

- cv2.IMREAD_COLOR：1：(H, W, 3)：BGR
- cv2.IMREAD_GRAYSCALE：0：(H, W)
- cv2.IMREAD_UNCHANGED：-1：(H, W, 4)：BGRA

imwrite 函数原型如下，对第三个参数稍加解释：

参考：[CSDN](https://www.cnblogs.com/wal1317-59/p/13469451.html)

```python
cv2.imwrite(file, img[, num])
```

第三个参数可选，含义如下：

```python
cv2.imwrite("x.jpg", img, (cv2.IMWRITE_JPEG_QUALITY， 95))
cv2.imwrite("x.png", img, (cv2.IMWRITE_PNG_COMPRESSION, 3))
```

- cv2.IMWRITE_JPEG_QUALITY=1：JPEG 格式，0-100 的整数，表示图像质量，默认为 95。
- cv.IMWRITE_PNG_COMPRESSION=16：PNG 格式，0-9表示压缩级别，级别越高图像越小，默认值为 3。

### RGB 与 BGR 转换（cv2 默认使用 BGR 格式）

```
# 将srcBGR: uint8数组(H, W, 3)转换为destRGB: uint8数组(H, W, 3)
destRGB = cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)
```

### 在图像上绘制

#### 画点（实际上是在画圈）

```python
# (x, y)为圆心的坐标, 需为整数
# 2为半径, (0, 0, 255)为圆的颜色BGR, 调用后img本身产生了变化
cv2.circle(img, (x, y), 2, (0, 0, 255), thickness=2)
```

#### 添加文字（非英文会乱码）

```python
cv2.putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
# org是左下角坐标, font是字体, fontScale是字体
cv2.putText(image, "mark", (100, 200), cv2.FONT_HERSHEY_COMPLEX, 5, (0, 0, 255), 6)
```



### 图像变换

```python
import sklearn.transform as trans
tform = trans.SimilarityTransform()
src = np.array([[100, 100], [200, 50], [200, 200], [100, 150]])
dst = np.array([[60, 70], [80, 70], [80, 100], [60, 100]])
tform.estimate(src, dst)
M = tform.params  # 3*3数组, 最后一行为[0, 0, 1]

src3d = np.ones(3, src.shape[0])
src3d[:-1, :] = src.T
# M @ src3d ~= dst
# M = 
# [[a0, -b0, a1],
#  [b0,  a0,  b1],
#  [0,   0,   1]]
```

变换公式为（`tform.estimate` 实际上就是在估计 `M`）：
$$
\begin{align}
dst_x &= a_0*src_x-b_0*src_y+a_1\\
&=s*[src_x*cos(\theta)-src_y*sin(\theta)]+a_1\\
dst_y &= b_0*src_x+a_0*src_y+b_1\\
&=s*[src_x*sin(\theta)+src_y*cos(\theta)]+b_1
\end{align}
$$
即：
$$
\begin{bmatrix}
dst_x\\dst_y\\0
\end{bmatrix}
=
\begin{bmatrix}
a_0&-b_0&a_1\\
b_0&a_0&b_1\\
0&0&1
\end{bmatrix}
\begin{bmatrix}
src_x\\src_y\\0
\end{bmatrix}
$$
#### cv2 图像旋转（待补充）

### 图像修复

cv2.inpaint

## OpenCV C++ API

### windows VS2017 + OpenCV C++

https://sevenold.github.io/2019/01/opencv-setup/
