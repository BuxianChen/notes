# TODO LIST

## Papers

- [PyTorch Distributed: Experiences on Accelerating Data Parallel Training](https://arxiv.org/pdf/2006.15704.pdf): Pytorch DDP 实现
- [Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation](https://arxiv.org/pdf/2004.09602.pdf): 量化(结合微信收藏看)
- [Learning to Navigate for Fine-grained Classification](https://arxiv.org/pdf/1809.00287.pdf): 细粒度分类
- [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/pdf/2005.08100.pdf): 语音conformer(视觉也有一篇重名了)
- [Pushing the Limit of Semi-Supervised Learning for Automatic Speech Recognition](https://arxiv.org/pdf/2010.10504)

## Python Tools

wandb：类似于 Tensorboard 的东西，待探索这类可视化工具对做实验是否有实际价值

argparse：

- metavar

- `mmcv/utils/config.py:DictAction` 继承自 `argparse.Action`，但似乎不能解析多重字典

配置解析：easydict.EasyDict，addict，yacs

~~numpy：~~

- ~~np.ndarray.flags, contiguaous~~

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

协程的概念与使用

~~pathlib：2021/11/3 完成（记录笔记推迟），官方文档[链接](https://docs.python.org/zh-cn/3/library/pathlib.html#correspondence-to-tools-in-the-os-module)，发现对 os.path 模块基本可以替换掉，但某些非 os.path 的函数例如 os.walk 没有很好的替代品~~

[h5py](https://docs.h5py.org/en/stable/)

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

YOLOv5 vs YOLOv6

- [x] [分布式训练的例子](https://github.com/tczhangzhi/pytorch-distributed)：2021/10/31 完成（还未测试成功）
- [ ] [GPU上进行数据预处理：DALI](https://github.com/NVIDIA/DALI)

~~[pdfminer](https://github.com/pdfminer/pdfminer.six)~~

~~[pdfplumber](https://github.com/jsvine/pdfplumber)~~

[wenet](https://github.com/wenet-e2e/wenet): C++ runtime
[faiss](https://github.com/facebookresearch/faiss): C++ and CUDA implement for KNN
[mmdeploy](https://github.com/open-mmlab/mmdeploy): 了解怎么将模型导出

## CS Tools

docker 使用及 Dockfile 的编写，“轻量”且“标准”容器的构建

vscode 推荐配置摸索与整理

shell 脚本学习与笔记完善

vim 的基本使用与插件

浏览器开发者工具的使用

~~pdb 与 pudb~~

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

~~JPG 与 PNG 格式详解（可以看看冈萨雷斯的书上是否有记录）~~

- pytorch 自动求导实现：https://colab.research.google.com/drive/1VpeE6UvEPRz9HmsHh1KS0XxXjYu533EC


## Blog Update Plan

- Jekyll、html/css/javascript 简介
- Swin Transformer
- HMM 语音识别模型（含 EM 算法）
- MFCC 具体解释


## Link

cmake:
  - [官方tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/index.html)
  - [上海交通大学视频](https://www.bilibili.com/video/BV14h41187FZ/)
  - [LLVM对CMake的介绍](https://llvm.org/docs/CMakePrimer.html#ft-view)
  - [pybind对已经用CMake方式组织的项目怎么集成](https://github.com/pybind/cmake_example)
  - [GitHub星数超过10k的cmake-examples](https://github.com/ttroy50/cmake-examples)

python 打包相关:
  - [python官方tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
    - 包含了一些关于上传至 PyPI 和 Test PyPI 的知识, 示例使用的是 `src` 结构与 `pyproject.toml` 的方式组织代码结构
  - [pytest文档里推荐的python项目组织形式](https://docs.pytest.org/en/7.2.x/explanation/goodpractices.html)
    - 打包配置文件: `pytest.ini`, `pyproject.toml`, `tox.ini`, `setup.cfg`(不推荐)
  - [python包的组织形式: src layout vs flatlayout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)
    - 结论: 如果希望作为python包来写, 推荐使用 src layout

python 协程相关:
  - [realpython文章](https://realpython.com/async-io-python/): 未消化
  - [解释协程与生成器关系的一篇博客](https://snarky.ca/how-the-heck-does-async-await-work-in-python-3-5/): 
  
pytest:
  - [import mode](https://docs.pytest.org/en/7.2.x/explanation/pythonpath.html#import-modes)
  - [关于项目源码与测试代码的组织形式](https://docs.pytest.org/en/7.2.x/explanation/goodpractices.html)
    - **项目源码推荐用 src layout, 测试代码推荐独立于项目源码, pytest 的 import mode 推荐 importlib(存疑, 似乎导致各个test脚本之间不能import), [参考博客](https://blog.ionelmc.ro/2014/05/25/python-packaging/#the-structure%3E)**
    - [setup.py, setup.cfg and pyproject.toml](https://ianhopkinson.org.uk/2022/02/understanding-setup-py-setup-cfg-and-pyproject-toml-in-python/)
  - 关于 setup.py, setup.cfg and pyproject.toml 以及 pytest 搭配用的实际例子可以参考 [transformers](https://github.com/huggingface/transformers) 代码库学习

## 2023 全年计划

已确定的

- 几种流行的大模型结构, 包括但不限于: OPT、BLOOM、LLAMA、GLM、GPTJ、GPTNEO、GPTNEOX、MOSS、RWKV 等
- 大模型训练所需的数据结构研究
- GPT-2 详解: 新增博客
- triton: 新增笔记
- CUDA: 新增笔记, 长期维护 (各种和 Nvidia 显卡相关的内容汇聚一下)
- torch benchmark, profiler 等性能测试: 新增博客
- bitandbytes: 新增博客
- huggingface optimum: 待定
- 自动微分: 结合陈天奇课程深入理解实现, 完善博客
- huggingface peft: 新增博客
- AutoGPTQ: 博客完善, 基本上需要结合前面所有的内容
- streamlet、gradio: 学习, 但不确定记录形式, 以及是否记录, 可能会写一些demo保存起来即可

暂时未定的
- 分布式训练: 结合李沐几篇论文精读以及torch自身的分布式内容整合为博客, 放至最后
- tvm/fastertransformer: 暂时不定
- torch.fx, torch.trace, torch.script, torch.compile: 暂时不定
