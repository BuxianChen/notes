# 利用原始下载的coco数据构建一个小型的标注数据集

import json
from collections import defaultdict
anno = json.load(open("../instances_val2017.json"))
a, b, c = 1, 1, 1
crowd, non_crowd, seg = 0, 0, 0

remain_data = defaultdict(list)

for item in anno["annotations"]:
    if (crowd, non_crowd, seg) == (a, b, c):
        break
    if item["segmentation"].__len__() > 1 and seg < c:
        seg += 1
        remain_data["seg"].append(item)
    if item["iscrowd"] == 0 and non_crowd < b:
        non_crowd += 1
        remain_data["non-crowd"].append(item)
    if item["iscrowd"] == 1 and crowd < a:
        crowd += 1
        remain_data["crowd"].append(item)
image_ids = [(key, _["image_id"]) for key, value in remain_data.items() for _ in value]
print(image_ids)  # 需要将涉及到的图片拷贝出来放在 images 文件夹下

new_anno = dict()
new_anno["info"] = anno["info"]
new_anno["licenses"] = anno["licenses"]
new_anno["categories"] = anno["categories"]
new_anno["images"] = [item for item in anno["images"] if item["id"] in list(zip(*image_ids))[1]]
new_anno["annotations"] = [_ for annos in remain_data.values() for _ in annos]

with open("mini_anno.json", "w") as fw:
    json.dump(new_anno, fw)