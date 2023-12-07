---
title: '通关注意力机制与Transformer任务：高频面试问题与解析分享'
collection: teaching
date: 2023-12-04
permalink: /teaching/20231204/cv-attention-awesome/
type: "大模型"
venue: "Attention, Transformer"
location: "City, Country"
tags:
  - 算法面试题
  - 深度学习面试题
  - 注意力机制
---

从视觉注意力机制到大模型Transformer架构，探索高频问题与深度解析。

# 01. 局部注意力如何实现

- 局部注意力则把加权求和的范围缩小到给定窗口大小的局部范围。于是，local Attention如何确定局部范围成为关键。第一种是采用自注意力机制来建模query对的关系。第二种是对query-independent的全局上下文建模。

- 参考链接：https://zhuanlan.zhihu.com/p/64988633

- Non-local block想要计算出每一个位置特定的全局上下文，但是经过训练之后，全局上下文是不受位置依赖的。三步实现注意力机制：1）全局attention pooling：采用1x1卷积和softmax函数来获取attention权值，然后执行attention pooling来获得全局上下文特征。2）特征转换：采用1x1卷积；3）特征聚合：采用相加操作将全局上下文特征聚合到每个位置的特征上。

- 卷积只能对局部区域进行context modeling，导致感受野受限制，而Non-local和SENet实际上是对整个输入feature进行context modeling，感受野可以覆盖到整个输入feature上，这对于网络来说是一个有益的语义信息补充。另外，网络仅仅通过卷积堆叠来提取特征，其实可以认为是用同一个形式的函数来拟合输入，导致网络提取特征缺乏多样性，而Non-local和SENet正好增加了提取特征的多样性，弥补了多样性的不足。

# 04. 介绍CBAM注意力

- 卷积模块的注意力机制模块，是一种结合了空间（spatial）和通道（channel）的注意力机制模块。
- 单就一张图来说，通道注意力，关注的是这张图上哪些内容是有重要作用的。平均值池化对特征图上的每一个像素点都有反馈，而最大值池化在进行梯度反向传播计算时，只有特征图中响应最大的地方有梯度的反馈。
- 将 CBAM 集成到 ResNet 中的方式是在每个block之间；

![Alt](/interview/cv-detection/cbam.png#pic_center=600x400)
