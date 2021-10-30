# 将标注可视化
import matplotlib.pyplot as plt
import json
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib
import os
import cv2
anno_file = "./annotations/mini_anno.json"
image_dir = "./images"
anno = json.load(open(anno_file))

def build_index(anno):
    index = dict()
    index["image_id2fname"] = {item["id"]: item["file_name"] for item in anno["images"]}
    return index

def plot_one_obj(single_anno, index):
    fig, ax = plt.subplots()
    image_path=os.path.join(image_dir, index["image_id2fname"][single_anno["image_id"]])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if isinstance(single_anno["segmentation"], list):
        patches = []
        for points in single_anno["segmentation"]:
            xys = np.array(points).reshape(-1, 2)
            patch = Polygon(xys)
            patches.append(patch)
        # collection=PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.2)
        collection=PatchCollection(patches, alpha=0.2, color=["red"]*len(patches))
        ax.add_collection(collection)
        ax.imshow(image)
    else:
        h, w = single_anno["segmentation"]["size"]
        mask = np.zeros(w*h, dtype=np.uint8)
        begin = 0
        for i, num in enumerate(single_anno["segmentation"]["counts"]):
            if i % 2:
                mask[begin: begin+num] = 1
            begin += num
        x = np.zeros_like(image)
        mask = mask.reshape((w, h)).transpose()
        image[mask.astype("bool"), 2] *= 1
        x[..., 2] = mask * 255 * 3
        image[mask.astype("bool"), 2] //= 4
        ax.imshow(image)
    return fig

index = build_index(anno)
fig = plot_one_obj(anno["annotations"][0], index)
#fig.show()