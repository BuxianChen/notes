## COCO 标注格式

[参考链接](https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch)

coco的目标检测任务的标注文件仅为一个json文件（含有所有图片的标注），形式如下：

```json
{
	"info": ... // 整个数据集的一些信息, 用处不大
	"licenses": ... // 数据所用到的开源协议列表, 不同图片的开源协议可能不一样
	"categories": ... // 数据里所有的物体种类
	"images": ... // 每张图片的基本信息
	"annotations": ... // 标注
}
```

**info**

```json
"info": {
    "description": "COCO 2017 Dataset",
    "url": "http://cocodataset.org",
    "version": "1.0",
    "year": 2017,
    "contributor": "COCO Consortium",
    "date_created": "2017/09/01"
},
```

**license**

```json
"licenses": [
    {
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License"
    },
    {
        "url": "http://creativecommons.org/licenses/by-nc/2.0/",
        "id": 2,
        "name": "Attribution-NonCommercial License"
    },
    ...
] // 一共有9种许可证
```

**categories**

```json
"categories": [
	{"supercategory": "person", "id": 1, "name": "person"},
  	{"supercategory": "vehicle", "id": 2, "name": "bicycle"},
  	...
],  // 一共有80种类别, 但id存在间段的情况
```

**images**

```json
"images": [
    {
        "license": 4,
        "file_name": "000000397133.jpg",
        "coco_url": "http://images.cocodataset.org/val2017/000000397133.jpg",
        "height": 427,
        "width": 640,
        "date_captured": "2013-11-14 17:02:52",
        "flickr_url": "http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg",
        "id": 397133  // 与filename没有必要一致
    },
    {
        "license": 1,
        "file_name": "000000037777.jpg",
        "coco_url": "http://images.cocodataset.org/val2017/000000037777.jpg",
        "height": 230,
        "width": 352,
        "date_captured": "2013-11-14 20:55:31",
        "flickr_url": "http://farm9.staticflickr.com/8429/7839199426_f6d48aa585_z.jpg",
        "id": 37777
    },
    ...
]
```

**annotations**

```json
"annotations": [
    {
        "segmentation": [[510.66,423.01,511.72,420.03,...,510.45,423.01]],  // 多边形的标注方式
        "area": 702.1057499999998,
        "iscrowd": 0,
        "image_id": 289343,
        "bbox": [473.07,395.93,38.65,28.67], // 左上角的宽，左上角的高，宽，高
        "category_id": 18,
        "id": 1768
    },
    ...
    {
        "segmentation": {
            "counts": [179,27,392,41,…,55,20], // RLE 的标注方式
            "size": [426,640]
        },
        "area": 220834,
        "iscrowd": 1,
        "image_id": 250282,
        "bbox": [0,34,639,388],
        "category_id": 1,
        "id": 900100250282
    }
]
```

annotations 是一个列表，每个列表代表一个标注（即一个物体）。各字段的含义为

- segmentation：表示实例分割标签，标注形式后面再详述。
- area：物体的面积
- iscrowd：若取值为 1 表示该条标注为一群物体，标注形式为 RLE 格式，会采用这种标注形式的例子为：叠在一起的书本，人群；若取值为 0 ，则表示该条标注代表一个轮廓清晰的物体，标注形式为多边形格式。
- image_id：图片ID，与 images 中的 id 字段相对应
- bbox：目标检测标签，四个数字依次代表：左上角的 $$x$$ 坐标，坐上角的 $$y$$ 坐标，目标框的宽，目标框的高。其中 $$x$$ 轴代表图片的上边缘，$$y$$ 轴代表图片的左边缘，原点位于左上角，四个数字均为绝对坐标（像素个数），但可能会有小数出现。
- category_id：物体类别，与 categories 中的 id 字段对应
- id：标注唯一性标识，无具体含义

注意：

- 标注信息均为像素位置的绝对值，注意像素位置，面积等标注信息可以是小数。
- annotations 列表是无序的，即同一张图片里的实例标注信息在 annotations 列表中不一定是连续的

实例分割标签分为两种：

- `iscrowd=0`：表示该实例为单个物体，segmentation 字段为一个列表，列表中的每个元素代表着一个多边形。例如：

  ```json
  {
  	"segmentation": [
  			[2, 134, 100, 134, 100, 200, 50, 270, 2, 200],
  			[102, 134, 150, 200, 102, 200]
  		]  // 连续的两个数字为一组代表一个顶点的坐标
  }
  ```

  代表该物体由一个凹五边形与一个三角形构成：

  ```
  (x, y)
  [(2, 134)->(100, 134)->(100, 200)->(50, 270)->(2, 200)->(2, 134)]
  [(102, 134)->(150, 200)->(102, 200)->(102, 134)]
  ```

- `iscrowd=1`：表示该实例为一群物体，segmentation 字段形式如下，counts 为 RLE 格式的mask。

  ```
  {
      "segmentation": {
      "counts": [0,179,27,392,...,41,55,20],
      "size": [240, 320]
  },
  ```

  `size` 字段表示 `[height, width]`，表示图片大小（该标注对应图片大小），而 `counts` 字段表示逐像素的 mask，mask 的形状为 `[height, width]`（在这个例子中为 `[240, 320]`），mask 为 1 表示该像素值属于物体，mask 为 0 表示该像素不属于物体。`counts` 的具体含义为：将 mask 拉直后（按列的方式拉直，即先取第一列，紧接着取第二列，以此类推），依次出现了 0 个 0，179 个 1，27 个 0，392 个 1 等等。

  备注：counts 的第一个元素一定是数有多少个 0，counts 的长度可以是奇数也可以是偶数，因此 counts 的最后一个元素可能是数有多少个 0，也可能是数有多少个 1。一个简易的转换代码如下：

  ```python
  def coco_mask2rle(mask):
      # mask (np.array): (height, width)
      # returns: rel (list)
      cur, count, rle = 0, 0, []
      for _ in mask.transpose().reshape(-1):
          if _ != cur:
              rle.append(count)
              count, cur = 1, _
          else:
              count += 1
      rle.append(count)
      return rle
  ```

一个简易的标注可视化代码参见 [show_mask.py](../.gitbook/assets/coco/show_mask.py)。（仅供理解，不要重复造轮子:blush:）

### pycocotools

**安装**

linux参考[官方源码](https://github.com/cocodataset/cocoapi)的说明即可。

windows 下需要安装第三方改写的包，如下：

```
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```

**简介**

```python
# image_id: 图像id, cat_id: 类别id, anno_id: 标注id 
class COCO: #全部函数如下
	def __init__(self, annotation_file=None):
		# 大体上是读取标注文件
        # ...
        self.createIndex()
    def createIndex(self):
        # 建立索引, 无返回
        pass
    def info(self):
        # 打印"info", 无返回
        pass
    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        # imgIds为指定的图像id列表, catIds为指定的标签id列表, areaRng为最小的面积阈值, iscrowd用于指定只返回iscrowd=0或1的标注
        # 返回满足条件的anno_id列表
    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        # 在json的categories字段中筛选: catNms用于指定"name"列表, supNms用于指定"supercategory"列表, catIDs用于指定"id"列表
        # 返回满足条件的cat_id列表
    def getImgIds(self, imgIds=[], catIds=[]):
        # 至少有catIds中的至少一个类别的imgIds中的图片
        # 返回满足条件的image_id列表
    def loadAnns(self, ids=[]):
        # ids表示anno_id列表，用于返回这些标注信息
        # 返回例子为：
        """
        [{"segmentation": [...], "bbox": [...], "id": 111, area": 112.32, ...},
        {"segmentation": [...], "bbox": [...], "id": 1123, "area": 1121.32, ...},
        ...]
        """
    def loadCats(self, ids=[]):
        # ids表示cat_id列表，用于返回类别信息
        # 返回例子为：
        """
        [{"id": 1, "name": "car", "supercategory": "car"},
        {"id": 2, "name": "cat", "supercategory": "animal"},
        ...]
        """
    def loadImgs(self, ids=[]):
        # ids表示image_id列表，用于返回图像信息
        # 返回例子为：
        """
        [{"id": 1, "file_name": "000100.jpg", "height": 100, "width": 120, ...},
        {{"id": 23, "file_name": "0001100.jpg", "height": 110, "width": 320, ...}},
        ...]
        """
    def showAnns(self, anns, draw_bbox=False)
    	"""作图: 不会读取相应的图片, 只对标注区域打阴影, 无返回值"""
    def loadRes(self, resFile):
        """待补充"""
    def download(self, tarDir = None, imgIds = [] ):
        # 下载数据
    def loadNumpyAnnotations(self, data):
        """
        Convert result data from a numpy array [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
        :param  data (numpy.ndarray)
        :return: annotations (python nested list), 例子：
        
        [{
            'image_id'  : imageID,
            'bbox'  : [x1, y1, w, h],
            'score' : score,
            'category_id': class,
        }]
        """
    def annToRLE(self, ann):
        """待补充"""
    def annToMask(self, ann):
        """待补充"""
```

## 数据增强

### Mixup

将两张图片逐像素相加，标签按比例分配

```python
# x1: (H, W, 3), x2: (H, W, 3)
# y1: onehot, y2: onehot
x = (alpha)*x1 + (1-alpha)*x2
label = alpha*y1 + alpha*y2
```

### Cutout

剪裁图像的一部分，标签不变

```python
# mask (L, 2), x: (H, W, 3), y: int
x[mask] = 0
label = y
```

### CutMix

剪裁的部分用另一张图的一部分替换，标签按比例分配

```python
# x1: (H, W, 3), x2: (H, W, 3)
# mask: (H, W)
# y1: onehot, y2: onehot
x = mask * x1 + (1 - mask) * x2
alpha = mask.sum() / mask.numel()
label = alpha * y1 + alpha * y2
```

## 常用库

### 图像格式

**图像保存尽量使用 `.png` 格式，`jpg` 格式是有损压缩的**

```python
import cv2
x = cv2.imread("b.jpg")  # 无论原始图片是jpg格式还是png格式, 保存成png格式总是无损的
cv2.imwrite("c.png", x)
y = cv2.imread("c.png")
(x == y).all()  # True
```

```python
import cv2
x = cv2.imread("b.png")
cv2.imwrite("c.jpg", x)
y = cv2.imread("c.jpg")  # 无论原始图片是jpg格式还是png格式, 保存成jpg格式总是有损的
(x == y).all()  # True
```

### OpenCV

[跳转](./opencv/opencv.md)

### PIL

```python
from PIL import Image
im = Image.open("./temp.jpg")
# print(im.format, im.size, im.mode)  # im.size=(width, height)
im_arr = np.array(im)
im_arr.shape  # (height, width)
im_arr.dtype  # np.uint8
out_im = Image.fromarray(im_arr.astype(np.uint8))  # 注意必须先转为像素值在0~255之间的uint8类型才能转换为Image对象
out_im.save("1.jpg")
```

PIL 默认的图像格式为 RGB，图像形状为 (H, W, 3)

CV2 默认的图像格式为 BGR，图像形状为 (H, W, 3)

mxnet 默认的图像读取格式为 RGB，图像形状为 (H, W, 3)

pytorch 默认的图像格式（网络输入）为 RGB，形状为 (B, 3, H, W)

tensoflow 默认的图像格式（网络输入）为 RGB，形状为 (B, H, W, 3)

```python
# 例如有如下错误:
from PIL import Image
import cv2
img_cv2 = cv2.imread("./images/unmasked.jpg")  # img_cv2(np.array): (H,W,3), BGR
img_pil = Image.fromarray(img_cv2)  # img_pil(Image), 默认认为img_cv2的RGB存储的
img_pil.save("pil.jpg")  # 保存的图片与原始图片颜色通道不一样
```

#### 利用 PIL 在图上写文字

由于 cv2 不支持中文文字，可以利用 PIL 实现

```python
from PIL import Image, ImageDraw, ImageFont
txt_mask = Image.new("RGB", (224, 224), (0, 0, 0))
txt_mask_draw = ImageDraw.Draw(txt_mask, mode="RGB")
font = ImageFont.truetype(font=r"C:\Windows\Fonts\Arial.ttf", size=24)
txt_mask_draw.text((100, 50), "font", (0, 0, 255))  # (100, 50)为左上角坐标
txt_mask.save("a.png")
```

备注：`PIL.ImageDraw.Draw.text` 函数的文字位置参数指的是文字的左上角坐标，而 `cv2.putText` 函数中的文字位置参数默认为文字的左下角坐标。

### base64

base64格式一般用于网络传输

```
disk -> byte: open("rb"), read
byte -> disk: open("wb"), write

byte -> base64: base64.b64encode
base64 -> byte: base64.b64decode
```

读取磁盘图片，不进行任何解码操作。（适用于任何文件，相当于拷贝文件）

```python
with open("a.jpg", "rb") as fr:
    b = fr.read()
with open("b.jpg", "wb") as fw:
    fw.write(b)
```

网络传输情形

```python
# with open("a.jpg", "rb") as fr:
    # b = fr.read()
# b64 = base64.b64encode(b)
# 网络传输一般从这里开始(接收到的是base64格式数据): 先将base64格式转为byte再保存
b = base64.b64decode(b64)
with open("b.jpg", "wb") as fw:
    fw.write(b)
```

### lmdb

lmdb 是一个 key-value 数据库，使用 Python 安装后即可使用。可用于存储多张图片数据。保存的方式为

```
train
  - data.mdb
  - lock.mdb
```

将图片从磁盘写入 lmdb 数据库的例子

```python
import lmdb
env = lmdb.open("./train", map_size=1024*1024*640)  # 单位为Byte, 即640M
txn = env.begin(write=True)
for image_name in image_names:
    with open(image_name, "rb") as fr:
        bs = fr.read()  # byte
    txn.put(key=image_name.encode("utf-8"）, value=bs) 
env.close()
```

将图片从上述 lmdb 数据库读出并转换为 cv2 的格式：

```python
import lmdb
env = lmdb.open("./train")
txn = env.begin()
# 遍历获取
for key, value in txn.cursor():
	image_name = key.decode(encoding="utf-8")
    image = cv2.imdecode(np.frombuffer(value, np.uint8), -1)
# 获取单条数据
bs = txn.get("xx.jpg".encode("utf-8"))
image = cv2.imdecode(np.frombuffer(bs, np.int8), -1)
env.close()
```

### mxnet: recordio

#### 细节与原理

mxnet 中推荐使用如下数据格式进行文件 IO，典型应用场景是原始数据为十万张图片以及相应的标签，mxnet 提供了相应的工具将所有的图片与标签压缩到一个后缀名为 `.rec` 的文件中，例如：

```
<dataname>.rec
```

`.rec` 文件的实际存储为二进制形式，（猜测）实际存储格式为：

```
<第一条数据的byte数><第一条数据的实际内容>
<第二条数据的byte数><第二条数据的实际内容>
...
```

其中字节数目占用 8 个字节的空间，而每条数据的实际内容所占的字节数不定长。

**mx.recordio.MXRecordIO**

```python
mxnet.recordio.MXRecordIO(uri, flag)  # flag可以取值为"w"或"r"
```

这个类只需要一个 `.rec` 文件作为输入，只处理最基本的读写操作，如下：

```python
record = mx.recordio.MXRecordIO('tmp.rec', 'w')
# <mxnet.recordio.MXRecordIO object at 0x10ef40ed0>
for i in range(3):
    record.write('record_%d'%i)
record.close()
record = mx.recordio.MXRecordIO('tmp.rec', 'r')
for i in range(3):
    item = record.read()
    print(item)
# record_0
# record_1
# record_2
record.close()
```

注意：由于每行数据所占的字节数是不一样的，所以 `MXRecordIO` 类只支持顺序读取，而不能进行随机读取（例如直接读取第 102 条数据）。

**mx.recordio.MXIndexedRecordIO**

```python
mxnet.recordio.MXIndexedRecordIO(idx_path, uri, flag, key_type=int)
```

这个类的主要作用是支持随机读写，因此还需要一个映射表用于指示每条数据的起始位置。需要两个文件作为输入，例如：

```
data.idx
data.rec
```

其中 `data.idx` 为一个文本文件，其内容大致为（分割符为制表符）：

```
1	0
2	5768
3	12520
4	19304
```

每行数据的第一个数字表示下标（即之后所述的 `read_idx(idx)` 中的 `idx`），第二个数字表示该下标对应的数据的起始位置。

小细节：`.rec` 文件中的字节对齐

```python
record = mx.recordio.MXRecordIO("data.rec", 'r')
for _ in range(3):
    x = record.read()  # x的类型为字节
    print(len(x))  # 依次为5760, 6742, 6773
```

注意实际存储时，每行的实际数据的字节数会对齐到 8 的倍数，因此实际存储时的存储如下

```
8byte 5760byte
8byte 6742byte 2byte(padding)
8byte 6773byte 3byte(padding)
```

`MXIndexedRecordIO` 的实际使用例子如下：

```python
for i in range(5):
    record.write_idx(i, 'record_%d'%i)
record.close()
record = mx.recordio.MXIndexedRecordIO('tmp.idx', 'tmp.rec', 'r')
record.read_idx(3)
record_3
```

**mx.recordio.pack/unpack/pack_img/unpack_img**

mxnet 中针对图像数据的 `.rec` 文件格式做了一些约定，当 `.rec` 文件的存储满足这些约定时，可以调用四个函数进行数据处理。以下 `header` 表示一个 `IRHeader` 对象，`s` 表示字节（即调用`MXRecordIO` 的 `read` 方法得到的东西），而 `img` 表示的是一个形状为 `(H, W, 3)` BGR 格式的三维数组。

```python
pack(header, s) -> s
unpack(s) -> header, s
pack_img(header, img, quality=95, img_fmt='.jpg') -> s
unpack_img(s, iscolor=-1) -> header, img
```

`IRHeader` 实际上就是一个 `namedtuple`，定义如下：

```
IRHeader = namedtuple('HEADER', ['flag', 'label', 'id', 'id2'])
```

- flag 是一个整数，可以自由根据需要设置
- label 是一个浮点数或浮点数组，代表标签
- id 是每条记录的唯一 id
- id2 一般设置为 0 即可

**工具**

待补充

`mxnet/tools/im2rec.py` 用于生成 `.rec` 格式的数据

#### 例子

**一个一般用法的例子**

```python
import mxnet as mx
import cv2
import numpy as np
import pickle

writer = mx.recordio.MXIndexedRecordIO("x.idx", "x.rec", "w")
for i, name in enumerate(["000001", "000002"]):
    header = mx.recordio.IRHeader(0, float(i), i*2, 0)
    img = cv2.imread(f"test_data/{name}.jpg")
    s = mx.recordio.pack_img(header, img, quality=95, img_fmt='.jpg')  # BGR
    writer.write_idx(i*2, s)
header = mx.recordio.IRHeader(0, np.array([1., 2.]), 10, 0)
b = pickle.dumps([1, 2, "dataname"])
s = mx.recordio.pack(header, b)
writer.write_idx(20, s)
writer.close()

reader = mx.recordio.MXIndexedRecordIO("x.idx", "x.rec", "r")
print(reader.keys)  # [0, 2, 20]
s = reader.read_idx(0)
header, img = mx.recordio.unpack(s)
img = mx.image.imdecode(img).asnumpy()  # RGB
img = np.ascontiguousarray(img[:, :, ::-1])
cv2.imwrite("000001.jpg", img)
print(header)

s = reader.read_idx(20)
header, content = mx.recordio.unpack(s)
print(pickle.loads(content))
print(header)

reader.close()
```

**一个实际例子：**

生成recordio形式的数据，参考[insightface](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/data/rec_builder.py)

```python
import pickle
import numpy as np
import os
import os.path as osp
import sys
import mxnet as mx


class RecBuilder():
    def __init__(self, path, image_size=(112, 112)):
        self.path = path
        self.image_size = image_size
        self.widx = 0
        self.wlabel = 0
        self.max_label = -1
        assert not osp.exists(path), '%s exists' % path
        os.makedirs(path)
        self.writer = mx.recordio.MXIndexedRecordIO(os.path.join(path, 'train.idx'), 
                                                    os.path.join(path, 'train.rec'),
                                                    'w')
        self.meta = []

    def add(self, imgs):
        #!!! img should be BGR!!!!
        #assert label >= 0
        #assert label > self.last_label
        assert len(imgs) > 0
        label = self.wlabel
        for img in imgs:
            idx = self.widx
            image_meta = {'image_index': idx, 'image_classes': [label]}
            header = mx.recordio.IRHeader(0, label, idx, 0)
            if isinstance(img, np.ndarray):
                s = mx.recordio.pack_img(header,img,quality=95,img_fmt='.jpg')
            else:
                s = mx.recordio.pack(header, img)
            self.writer.write_idx(idx, s)
            self.meta.append(image_meta)
            self.widx += 1
        self.max_label = label
        self.wlabel += 1


    def add_image(self, img, label):
        #!!! img should be BGR!!!!
        #assert label >= 0
        #assert label > self.last_label
        idx = self.widx
        header = mx.recordio.IRHeader(0, label, idx, 0)
        if isinstance(label, list):
            idlabel = label[0]
        else:
            idlabel = label
        image_meta = {'image_index': idx, 'image_classes': [idlabel]}
        if isinstance(img, np.ndarray):
            s = mx.recordio.pack_img(header,img,quality=95,img_fmt='.jpg')
        else:
            s = mx.recordio.pack(header, img)
        self.writer.write_idx(idx, s)
        self.meta.append(image_meta)
        self.widx += 1
        self.max_label = max(self.max_label, idlabel)

    def close(self):
        with open(osp.join(self.path, 'train.meta'), 'wb') as pfile:
            pickle.dump(self.meta, pfile, protocol=pickle.HIGHEST_PROTOCOL)
        print('stat:', self.widx, self.wlabel)
        with open(os.path.join(self.path, 'property'), 'w') as f:
            f.write("%d,%d,%d\n" % (self.max_label+1, self.image_size[0], self.image_size[1]))
            f.write("%d\n" % (self.widx))
```

读取上述recordio形式的数据，参考[insightface](https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/dataset.py)

```python
import numbers
import os

import mxnet as mx
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)
```

### mxnet

mxnet 与 pytorch 一样，对于卷积算法，默认情况下会开启自动搜索最快算法的功能。如果需要关闭此功能，需要进行如下操作（参考 [github-issue](https://github.com/apache/incubator-mxnet/issues/8132)）：

- 首先将模型保存文件 `*.json` 中的如下键值对修改：

  ```
  "cudnn_tune": "limited_workspace"
  修改为
  "cudnn_tune": "none"
  ```

  备注：mxnet 模型保存形式为两个文件：`*.json` 与 `*.params`。

- 运行时需修改环境变量

  方案一：

  ```bash
  $ export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
  ```

  方案二：在 python 代码中添加

  ```python
  os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
  ```

  

## 人脸识别任务

### 数据集及评价标准

#### LFW 数据集

> ​       LFW数据集共有13233张人脸图像，每张图像均给出对应的人名，共有5749人，且绝大部分人仅有一张图片。每张图片的尺寸为250X250，绝大部分为彩色图像，但也存在少许黑白人脸图片。
> ​       LFW数据集主要测试人脸识别的准确率，该数据库从中随机选择了6000对人脸组成了人脸辨识图片对，其中3000对属于同一个人2张人脸照片，3000对属于不同的人每人1张人脸照片。测试过程LFW给出一对照片，询问测试中的系统两张照片是不是同一个人，系统给出“是”或“否”的答案。通过6000对人脸测试结果的系统答案与真实答案的比值可以得到人脸识别准确率
> ————————————————
> 版权声明：本文为CSDN博主「姚路遥遥」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
> 原文链接：https://blog.csdn.net/Roaddd/article/details/114221618

#### Megaface 数据集

**测试方法**

Cumulative Match Characteristics (CMC) curve：将待

### 人脸关键点

参考链接：[CSDN](https://blog.csdn.net/u013841196/article/details/85720897)

![](../.gitbook/assets/cv/face_keypoint_68.jpg)

68 个关键点如上图所示，转换为 5 个关键点的方式为

```python
landmark5[0] = (landmark68[36] + landmark68[39]) / 2  # 右眼
landmark5[1] = (landmark68[42] + landmark68[45]) / 2  # 左眼
landmark5[2] = landmark68[30] # 鼻子
landmark5[3] = landmark68[48] # 右嘴角
landmark5[4] = landmark69[54] # 左嘴角
```

### insightface

`recognition/_tools_/mask_renderer.py`：自动生成口罩遮挡的人脸

训练/验证数据集：

例如：faces_umd.zip

```
agedb_30.bin  # 1:1的验证数据（训练时使用）
cfp_fp.bin
lfw.bin
property  # 类别数, 112, 112
train.idx  # 索引,train.rec中哪些byte是一个图片
train.rec  # mxnet二进制文件,包含图片及类别
```



测试数据集

例如：IJBC

```
IJBC 
    - loose_crop
        - xxx.jpg  # 1.jpg~469375.jpg
    - meta
        - ijbc_face_tid_mid.txt  # 469375行
        - ijbc_name_5pts_score.txt  # 469375行
        - ijbc_template_pair_label.txt  # 15658489行
```

ijbc_face_tid_mid.txt 文件内容解释：第一项为图片名，第二项表示该图片是哪个人，第三个数字是视频 ID。注意：同一个人可能有多个ID，例如

```
1.jpg 1 69544
2.jpg 1 3720
...
469375.jpg 187955 111105
```

ijbc_name_5pts_score.txt 文件内容解释：第一项为文件名，后面连续 10 个浮点数两个一组分别为 5 个关键点的 $$(x,y)$$ 绝对像素位置坐标，最后一个数字是一个接近 1 的分数（该图片是人脸的置信度）

ijbc_template_pair_label.txt 文件内容解释：前两个为人物 ID，如前面所述，两个 ID 可能对应的是同一个人，由第三个数字指示。

```
1 11065 1
1 11066 1
...
171707 185794 0
```

不同的人物 ID 数有 23124 个，但不同的 ID 有可能对应到同一个人。并且人物 ID 与视频 ID 都**不是**从 1 开始连续编号。

#### IJBC 测试数据集

每个 ID 对应多张图片，每个人物对应多个 ID。测试时判断两个 ID 是否是同一个人。测试标签例子如下：

```
person_id_1	person_id_2	1  // 表示这两个ID是同一人
person_id_3	person_id_4	0  // 表示这两个ID不是同一人
```

范式如下：训练一个网络，输入是人脸图片，输出是一个 512 维的特征。测试时，将所有属于同一个 person_id 的图片的输出特征取平均。对于每一条测试数据，求两个 person_id 对应特征的余弦距离，设定一个阈值来判断这两个 person_id 是否为同一个人。





度量指标

1: 1 验证：

- ROC & TAR@FAR

$$
TAR=\frac{TP}{TP+FN}=\frac{TP}{P}\\
FAR=\frac{FP}{FP+TN}=\frac{FP}{N}
$$

一般而言，负样本对的数目会比较大，因此 FAR 会很小，性能指标会使用 TAR@FAR 的形式表示：通过调整阈值，使得 FAR 值达到指定的要求，统计此时的 TAR，例如：$$0.95@10^{-6}$$，表示调整相似度阈值使得 FAR 为 $$10^{-6}$$ 时，TAR的值为 0.95。备注：随着阈值的提升，TAR 与 FAR 同时下降。

另一种度量指标是对不同的相似度阈值描点：`(FAR, TAR)`，得到 ROC 曲线，计算 ROC 曲线下的面积。

LFW 准确率评估指标：将测试数据分为 K 组，每次选择其中一个组调整阈值使得准确率最高，在剩余的 K-1 个组上计算准确率。对每个组重复上述操作，得到均值与方差，例如：$$0.95\pm0.01$$。

1: n 识别：

Megaface 比赛评测方法：

使用 K 张图片做为底库（gallery set），测试样本为 N 个人物，每个人 M 张图片，这里 $$N=80, M=50, K=10^6$$，这 $$N\times M$$ 张图片称为 probe set。gallery set 中没有这 N 个人物的图片，每次测试都将一个人的一张照片放入到 gallery set 中，用剩余的 M-1 张图片与 $K+1$ 张图片计算相似度并排序，判断与这张图片的相似度大小是否排在 Rank-K 以内。



实验记录：

Probe set （facescrub）图片数量：不到 4000 张

MegaFace Gallery set 未去噪前图片数量：1027058



## 开源工具

### labelImg

标注目标检测框

### labelme

标注实例分割，语义分割，目标检测等任务

安装

```
pip install labelme
```



实例分割

```
labelme data_annotated --labels labels.txt --nodata --validatelabel exact --config '{shift_auto_shape_color: -2}'
```

- `data_annotated` 为存储图片的文件夹
- `--labels label.txt` 表示标签文件
- `--nodata` 表示生成的标注文件中不存储图片
- `--validatelabel exact` 表示不允许标注不在 `labels.txt` 中的标签
- `--config '{shift_auto_shape_color: -2}'`  的含义未知
- 操作界面里每个多边形需要给出标签，另外还可以选择给一个 `group_id`，适用于用多个多边形框住一个物体的情况
- 输出的标注形式是每张图片一个 Json 文件，格式为 labelme 包的格式

将标注格式转换为 coco 格式

```python
# labelme-4.5.12/examples/instance_segmentation/labelme2coco.py
./labelme2coco.py data_annotated data_dataset_coco --labels labels.txt
```

- `data_annotated` 文件夹中存放着上一步得到的一堆 Json 文件（labelme 包的格式）

- `data_dataset_coco` 为转换后的文件目录，转换后会得到

  ```
  data_dataset_coco
    - JPEGImages/  # 原始图片, 不包含没有被标注的文件
    - Visualization/  # 标注可视化结构
    - annotation.json  # coco格式的标注
  ```

### TextRecognitionDataGenerator

生成文本图片数据

版本号

```
commit id: 9cc441
```

使用方法

```
python run.py ...
```

- `--output_dir`

- `--input_dir`

- `-l en`：设置语言，必须有 `fonts/lantin` 文件夹，`-l cn` 必须有 `fonts/cn` 文件夹，文件底下都是 `.ttf` 文件。脚本运行的逻辑每张图片随机从这些字体中选择

- `-c 1000`：生成 1000 张小条图

- `-rs`，`-let`，`-num`，`-sym`：适用于生成随机文本进行生成的情况，用法略去

- `-w`，`-r`：不明含义，看字面意思是生成小条图中有多少个字符，但似乎总是会生成整个文本

  ```python
  parser.add_argument("-w", "--length", type=int, nargs="?", help="Define how many words should be included in each generated sample. If the text source is Wikipedia, this is the MINIMUM length",default=1)
  
  parser.add_argument("-r", "--random", action="store_true", help="Define if the produced string will have variable word count (with --length being the maximum)", default=False)
  ```

- `-f 32`：将小条图的高度设置为 32 像素

- `-t 4`：运行 `run.py` 脚本的线程数

- `-e jpg`：生成图片的后缀名

- `-k 20`：生成的文字逆时针 20 度，`-rk` 表示随机旋转 `[-20, 20]` 度角

- `-wk`：不明含义

- `-bl 3`：表示增加高斯模糊，高斯模糊的卷积核大小为 3，`-rbl` 表示随机适用 `[0, 3]` 的高斯模糊核

- `-b`：`-b 3` 表示背景图片从 `pictures` 文件夹里随机选择

- `-hw`：生成手写字符，需要 tensorflow 的训练模型，未测试

- `-na 0`：输出图片的命名方式为 `[TEXT]_[ID].[EXT]`

- `-d 2`：采用的 distorsion （图像扭曲）方式为 Cosine wave

- `-do`：`-d 0` 表示只对垂直方向做扭曲，`1` 表示只对水平方向做扭曲，`2` 表示对两个方向都做扭曲

- `-wd`：`-wd -1` 根据字符串长度自动确定小条图的长度（像素个数），`-wd 300`表示手动设定小条图的长度

- `-al 1`：设置对齐方式为中心对齐

- `-or 1`：设置文字方向为垂直，即每个字的方向都是正向的，但书写方向为从上自下

- `-tc #FF0000`：文字的颜色为红色

- `-sw 1.5`：字符间隔，测试发现似乎无效

源码分析

```
commit id: 989bcc7c (2021.9.20)
```

目录结构如下

```
ROOT
  - custom/  # 输出样例
  - samples/  # 输出样例
  - tests/  # 输出样例
  - trdg/ # 主要的代码
    - dicts/
    - fonts/
    - generators/
    - handwritten_model/
    - images/
    - texts/
    - __init__.py
    - background_generator.py
    - computer_text_generator.py
    - data_generator.py
    - distorsion_generator.py
    - handwritten_text_generator.py
    - run.py
    - string_generator.py
    - utils.py
  - .gitignore
  - .travis.yml
  - codecov.yml
  - Dockerfile
  - LICENSE
  - MANIFEST.in
  - README.md
  - requirements-hw.txt  # 如果需要生成手写体图片, 需要安装更多的包
  - requirements.txt
  - setup.cfg
  - setup.py
  - tests.py  # 使用unittest进行单元测试的代码
```

主要代码集中在 `trdg` 文件夹下，其入口为 `trdg/run.py` 的 `main` 函数，其实际逻辑主要是利用 `argparse` 模块从命令行获取控制输出的参数，利用多进程调用 `trdg/data_genearator.py` 中的类方法 `FakeTextDataGenerator.generate_from_tuple` 生成图片：

```python
# run.py核心代码
from multiprocessing import Pool
...
p = Pool(args.thread_count)
p.imap_unordered(
    FakeTextDataGenerator.generate_from_tuple,
    zip([i for i in range(0, string_count)], strings, ...))
p.terminate()
...
```

而这个函数内部将依次调用：

- `trdg/computer_text_generator.py:generate` 或 `trdg/computer_text_generator.py:generate`：分别适用于生成打印体与手写体。对于前者，实际上主要就是调用了 `PIL.Image.text` 方法生成文字图片以及文字区域的`mask`。
- `trdg/distorsion_generator.py:*`：`*` 是 `sin`、`cos`、`random` 中的一个。用于对前一步生成的图片进行扭曲
- `trdg/background_generator.py:*`：`*` 是 `gaussian_noise`、`plain_white`、`quasicrystal`、`image` 中的一个，用于生成背景图片，注意：`image` 函数是利用用户指定的图片作为背景
- 利用 `PIL.Image.paste` 方法将背景图片与扭曲过的图片融合，得到融合后的图片
- 利用 `PIL` 的一些方法加上一些模糊
