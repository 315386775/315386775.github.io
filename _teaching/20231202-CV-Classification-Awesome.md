---
title: '通关计算机视觉目标分类任务：百道高阶高频面试题'
collection: teaching
date: 2023-12-02
permalink: /teaching/20231202/cv-classification-awesome/
type: "Classification"
venue: "MLT, Projects"
location: "City, Country"
tags:
  - 算法面试题
  - 深度学习面试题
  - 计算机视觉检测
---

从大模型来回溯目标分类任务，高阶题目深度分析。

# 01. 视觉任务中的长尾问题的常见解决方案

- 两种基本方法：重采样、重加权，其中长尾分类最优的Decoupling算法依赖于2-stage的分步训练，特征提取backbone需要在长尾分布下学，而classifier又需要re-balancing的学。

- 但上述方法的问题是需要在训练/学习之前，了解“未来”将要看到的数据分布，这显然不符合人类的学习模式，De-confound-TDE

- 参考链接：https://zhuanlan.zhihu.com/p/259569655
