### COCO

[参考链接](https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch)

coco的标注文件仅为一个json文件（含有所有图片的标注）

```json
{
	"info": ... // 整个数据集的一些信息, 用处不大
	"licenses": ... // 数据所用到的开源协议列表, 不同图片的开源协议可能不一样
	"categories": ... // 数据里所有的物体种类
	"images": ... // 每张图片的基本信息
	"annotations": ... // 标注
}
```



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

```json
"categories": [
	{"supercategory": "person", "id": 1, "name": "person"},
  	{"supercategory": "vehicle", "id": 2, "name": "bicycle"},
  	...
],  // 一共有80种类别, 但id存在间段的情况
```

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

```json
"annotations": [
    {
        "segmentation": [[510.66,423.01,511.72,420.03,...,510.45,423.01]],  // 多边形的
        "area": 702.1057499999998, //
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

