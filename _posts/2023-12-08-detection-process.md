---
title: '大幅提升小目标检测性能：面向深度学习超大分辨率输入图像的通用处理框架'
date: 2023-12-08
permalink: /posts/2023/12/detection-process/
tags:
  - 目标检测
  - SAHI
  - Ultralytics
---

为了解决小目标检测问题，同时也可以解决超大分辨率图像的问题（如遥感图像、病理图像、医疗影像等），本文提出了一个在微调和推理阶段基于切片的通用框架。将输入图像划分为重叠的切片，对于小目标相对于输入网络的图像产生相对较大的像素区域。

![对比](/posts/blog1204/00.png)



YOLOv8 用法
======

新的[ultralytics包](https://pypi.org/project/ultralytics/)可以轻松地使用自定义数据训练 YOLO 模型并将其转换为 ONNX 格式以进行部署。

以下是 Python API 的示例：

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
results = model.train(data="coco128.yaml", epochs=3)  # train the model
results = model.val()  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
success = YOLO("yolov8n.pt").export(format="onnx")  # export a model to ONNX format
```


参考链接
======
- [YOLOV8稀疏量化部署](https://neuralmagic.com/blog/yolov8-detection-10x-faster-with-deepsparse-500-fps-on-a-cpu/)