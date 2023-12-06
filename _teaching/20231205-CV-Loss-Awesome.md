---
title: '模型部署系列：Test'
collection: teaching
date: 2023-12-05
permalink: /teaching/20231205/cv-loss-awesome/
type: "CV"
venue: "A, deeplearning"
location: "City, Country"
tags:
  - TensorRT
  - 量化
  - AI模型部署
---


- TensorRT 是硬件相关的, 不同显卡其核心数量、频率、架构、设计都是不一样的，TensorRT 需要对特定的硬件进行优化，不同硬件之间的优化是不能共享的。

- TensorRT 官网链接：https://developer.nvidia.com/tensorrt

TensorRT 为什么能跑那么快
======
TensorRT 执行五种类型的优化以提高深度学习模型的吞吐量。

![优化](/posts/blog1201/00.png)
