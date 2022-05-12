# TODO LIST

## Papers

- [PyTorch Distributed: Experiences on Accelerating Data Parallel Training](https://arxiv.org/abs/2006.15704)
- [Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation](https://arxiv.org/abs/2004.09602)
- [Learning to Navigate for Fine-grained Classification](https://arxiv.org/abs/1809.00287): 细粒度分类

## Python Tools

wandb：类似于 Tensorboard 的东西，待探索这类可视化工具对做实验是否有实际价值

argparse：

- metavar

- `mmcv/utils/config.py:DictAction` 继承自 `argparse.Action`，但似乎不能解析多重字典

配置解析：easydict.EasyDict，addict，yacs

numpy：

- np.ndarray.flags, contiguaous

Pytorch

- 提高显卡利用率
- 多卡训练官方教程
- Apex（timm中可以选择的并行方式）
- ~~基础分类的 Demo Code：Dataset，DataLoader，transformers（图像增强），Optimizers，Scheduler，梯度剪裁，Loss，DistributeDataParallel~~（已完成）
- Pytorch 笔记完善
- scatter/gather
  - mmcv/parallel/scatter_gather.py
  - srgocr/core/parallel/scatter_gather.py
  - torch/nn/parallel/scatter_gather.py
  - torch/distributed/distributed_c10d.py

多进程库的使用

- [ ] pathlib：2021/11/3 完成（记录笔记推迟），官方文档[链接](https://docs.python.org/zh-cn/3/library/pathlib.html#correspondence-to-tools-in-the-os-module)，发现对 os.path 模块基本可以替换掉，但某些非 os.path 的函数例如 os.walk 没有很好的替代品

## Deep Learning Papers and Corresponding Code

Transformer，ViT，Swin Transformer，DETR

目标检测：Cascade RCNN，HTC，YOLO

文字检测：PSE

OCR 文字识别：SAR

图网络与图像聚类结合的论文

剪枝蒸馏量化

## GitHub Projects

mmcv，mmdetection（目标：不使用开发模式安装mmdet来开发项目），mmOCR

- [ ] mmcv：
  - DataContainer（mmcv/parallel/data_container.py）
  - MMDistributedDataParallel（mmcv/parallel/distributed.py）
  - save_checkpoint（mmcv/runner/checkpoint.py）
  - BaseRunner（mmcv/runner/base_runner.py）
  - Hook（mmcv/runner/hooks/hook.py）

YOLOv5（弄清细节）

- [x] [分布式训练的例子](https://github.com/tczhangzhi/pytorch-distributed)：2021/10/31 完成（还未测试成功）
- [ ] [GPU上进行数据预处理：DALI](https://github.com/NVIDIA/DALI)

[pdfminer](https://github.com/pdfminer/pdfminer.six)

[pdfplumber](https://github.com/jsvine/pdfplumber)

## CS Tools

docker 使用及 Dockfile 的编写，“轻量”且“标准”容器的构建

vscode 推荐配置摸索与整理

shell 脚本学习与笔记完善

vim 的基本使用与插件

浏览器开发者工具的使用

pdb 与 pudb

git 换行符问题，参考[链接](https://adaptivepatchwork.com/2012/03/01/mind-the-end-of-your-line/)做笔记整理

missing semester 第5课（记录在shell and linux中即可）

- job control
- tmux
- dotfile 配置
- remote

## 学习计划

- Jekyll 模板的摸索与 GitHub Pages 的探索
- 《linux shell scripting cookbook (2ed)》
- missing semester 全部课程及作业
- CPython Internals (随缘)
- CUDA C 基础学习，oneflow、pytorch 选学

## 杂项

数据转换：COCO-YOLOv5互转，mxnet RECORD格式，lmdb格式

Python自带的数据库：sqlite3

代码文档自动生成：[MkDocs](https://www.mkdocs.org/)，使用例子：[Spektral](https://github.com/danielegrattarola/spektral/)

JPG 与 PNG 格式详解（可以看看冈萨雷斯的书上是否有记录）

## Blog Update Plan

- Jekyll、html/css/javascript 简介
- Swin Transformer
- HMM 语音识别模型（含 EM 算法）
- MFCC 具体解释
