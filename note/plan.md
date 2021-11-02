# TODO LIST

## Python Tools

wandb：类似于 Tensorboard 的东西，待探索这类可视化工具对做实验是否有实际价值

argparse：metavar

配置解析：easydict.EasyDict，addict，yacs

numpy：

- np.ndarray.flags, contiguaous

Pytorch

- 提高显卡利用率
- 多卡训练官方教程
- Apex（timm中可以选择的并行方式）
- 基础分类的 Demo Code：Dataset，DataLoader，transformers（图像增强），Optimizers，Scheduler，梯度剪裁，Loss，DistributeDataParallel
- Pytorch 笔记完善

多进程库的使用

- [ ] pathlib：2021/11/3 完成

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
- [ ] [GPU上进行数据预处理](https://github.com/NVIDIA/DALI)

[pdfminer](https://github.com/pdfminer/pdfminer.six)

[pdfplumber](https://github.com/jsvine/pdfplumber)

## CS Tools

docker 使用及 Dockfile 的编写，“轻量”且“标准”容器的构建

vscode 推荐配置摸索与整理

shell 脚本学习与笔记完善

vim 的基本使用与插件

浏览器开发者工具的使用

## 杂项

数据转换：COCO-YOLOv5互转，mxnet RECORD格式，lmdb格式

Python自带的数据库：sqlite3

代码文档自动生成：[MkDocs](https://www.mkdocs.org/)，使用例子：[Spektral](https://github.com/danielegrattarola/spektral/)

JPG 与 PNG 格式详解（可以看看冈萨雷斯的书上是否有记录）
