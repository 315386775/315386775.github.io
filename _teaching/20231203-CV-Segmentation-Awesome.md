---
title: '通关计算机视觉目标分割任务：百道高阶高频面试题'
collection: teaching
date: 2023-12-03
permalink: /teaching/20231203/cv-segmentation-awesome/
type: "Segmentation"
venue: "SAM, CV"
location: "City, Country"
tags:
  - 算法面试题
  - 深度学习面试题
  - 计算机视觉检测
---

从SAM来回溯目标分割任务，高阶题目深度分析

- TensorRT 是硬件相关的, 不同显卡其核心数量、频率、架构、设计都是不一样的，TensorRT 需要对特定的硬件进行优化，不同硬件之间的优化是不能共享的。

- TensorRT 官网链接：https://developer.nvidia.com/tensorrt

TensorRT 为什么能跑那么快
======
TensorRT 执行五种类型的优化以提高深度学习模型的吞吐量。

![优化](/posts/blog1201/00.png)
