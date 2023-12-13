---
title: '人工智能数据集可视化统计分析工具：快速了解你的数据集'
date: 2023-12-11
permalink: /posts/2023/12/dataset-analysis/
tags:
  - 数据可视化
  - 数据分析
  - 目标检测
---

Lightly Insights：可以轻松获取关于机器学习数据集基本洞察的工具，可以可视化图像数据集的基本统计信息，仅需提供一个包含图像和对象检测标签的文件夹，它会生成一个包含指标和图表的静态 HTML 网页。

## 特征

- 支持所有可以使用Labelformat包读取的对象检测标签格式。其中包括 YOLO、COCO、KITTI、PascalVOC、Lightly 和 Labelbox。
- 显示图像、对象和类别计数
- 分析有多少图像没有标签，并提供它们的文件名。
- 显示图像样本
- 显示图像和物体尺寸的分析
- 显示每个类的分析，包括对象大小、每个图像的计数、位置热图等。

## 示例报告

![Alt](/posts/blog1211/screenshot0.png#pic_center=600x400)
![Alt](/posts/blog1211/screenshot1.png#pic_center=600x400)
![Alt](/posts/blog1211/screenshot2.png#pic_center=600x400)

## 安装

```sh
pip install lightly-insights
```


## 用法

Lightly Insights 报告由 python 脚本生成。下面的示例使用PascalVOC 2007数据集。您可以按照示例下载它（~450MB）：

```sh
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtrainval_06-Nov-2007.tar
```

要运行 Lightly Insights，我们需要提供：

- 图片文件夹。在我们的例子中就是./VOCdevkit/VOC2007/JPEGImages。
- 物体检测标签。对于 PascalVOC，构造函数需要带有注释 ./VOCdevkit/VOC2007/Annotations和类列表的文件夹。

```python

from pathlib import Path
from labelformat.formats import PascalVOCObjectDetectionInput
from lightly_insights import analyze, present

# Analyze an image folder.
image_analysis = analyze.analyze_images(
    image_folder=Path("./VOCdevkit/VOC2007/JPEGImages")
)

# Analyze object detections.
label_input = PascalVOCObjectDetectionInput(
    input_folder=Path("./VOCdevkit/VOC2007/Annotations"),
    category_names=(
        "person,bird,cat,cow,dog,horse,sheep,aeroplane,bicycle,boat,bus,car,"
        + "motorbike,train,bottle,chair,diningtable,pottedplant,sofa,tvmonitor"
    )
)
od_analysis = analyze.analyze_object_detections(label_input=label_input)

# Create HTML report.
present.create_html_report(
    output_folder=Path("./html_report"),
    image_analysis=image_analysis,
    od_analysis=od_analysis,
)
```
要查看报告，请打开./html_report/index.html.


## 引用

[1] [lightly-insights](https://github.com/lightly-ai/lightly-insights) 

