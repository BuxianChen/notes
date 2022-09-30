
## 参考资料

- [pytorch-lightning 101](https://www.youtube.com/playlist?list=PLaMu-SDt_RB5NUm67hU2pdE75j6KaIOv2): 视频课程, 可以用来理解概念, 共 4 小节课程, 其中第 3 小节是 pytorch-lightning 的基本用法, 第 4 小节介绍了 pytorch-lightning 的实现细节
- [LightningLite](https://pytorch-lightning.readthedocs.io/en/latest/starter/lightning_lite.html): 轻量版的 pytorch-lighting, 目前(2022.9.29)并未完全成熟. 用于尽量做很少的代码改动, 快速将 pytorch 训练代码进行转换, 好处是可以很快地将写好的单 GPU 或 CPU 训练流程变得可以自动支持多卡训练, fp16 训练等.