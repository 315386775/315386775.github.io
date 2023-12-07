---
title: '通关深度学习损失与优化器任务：面试问题与项目实战深度解析'
collection: teaching
date: 2023-12-05
permalink: /teaching/20231205/cv-loss-awesome/
type: "Model"
venue: "Loss, Optimzer"
location: "City, Country"
tags:
  - 算法面试题
  - 深度学习面试题
  - 计算机视觉检测
---

深度学习模型构建最重要的损失函数与优化器部分，本文列举在企业面试中高频问题，在此基础上结合项目实战经验给予深度解析。

- TensorRT 是硬件相关的, 不同显卡其核心数量、频率、架构、设计都是不一样的，TensorRT 需要对特定的硬件进行优化，不同硬件之间的优化是不能共享的。

- TensorRT 官网链接：https://developer.nvidia.com/tensorrt

TensorRT 为什么能跑那么快
======
TensorRT 执行五种类型的优化以提高深度学习模型的吞吐量。

![优化](/posts/blog1201/00.png)
