### COCO

[参考链接](https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch)

coco的目标检测任务的标注文件仅为一个json文件（含有所有图片的标注）

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

#### **annotations**

```json
"annotations": [
    {
        "segmentation": [[510.66,423.01,511.72,420.03,...,510.45,423.01]],  // 多边形的
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
            "counts": [179,27,392,41,…,55,20],
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

- `iscrowd=1`：表示该实例为一群物体



```python
# mask可视化
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np

N = 5
patches = []

for i in range(1):
    polygon = Polygon([[0, 0], [1, 0.5], [1, 1], [0.5, 1], [0.5, 0.5]], True)
    patches.append(polygon)
fig, ax = plt.subplots()
p = PatchCollection(patches, alpha=0.5)
ax.add_collection(p)
plt.show()
```

