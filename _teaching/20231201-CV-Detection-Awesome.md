---
title: '通关计算机视觉检测任务：百道常见面试题深度解析'
collection: teaching
date: 2023-12-01
permalink: /teaching/20231201/cv-detection-awesome/
type: "Detection"
venue: "Anchor-Free, Anchor-Base"
location: "City, Country"
tags:
  - 算法面试题
  - 深度学习面试题
  - 计算机视觉检测
---


前述
======

本博客面向深度学习算法中常见的视觉检测任务，甄选百道常见面试题来进行深度解析，Mark.AI希望通过这些题目的整理和解析，能够为求职者提供一个系统、全面的面试准备资源，帮助你们在面试中表现出色，取得成功。该博客持续更新真实面试题，关注不迷路。


# 01. Yolov5的Foucs层和Passthrough层有什么区别

Focus层原理和PassThrough层很类似。它采用切片操作把高分辨率的图片（特征图）拆分成多个低分辨率的图片/特征图，即隔列采样+拼接。

原始的640 × 640 × 3的图像输入Focus结构，采用切片（slice）操作，先变成320 × 320 × 12的特征图，拼接（Concat）后，再经过一次卷积（CBL(后期改为SiLU，即为CBS)）操作，最终变成320 × 320 × 64的特征图。

![Alt](/interview/cv-detection/foucs.jpg#pic_center=300x200)

Focus层将w-h平面上的信息转换到通道维度，再通过3*3卷积的方式提取不同特征。采用这种方式可以减少下采样带来的信息损失 。

```python
class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
 
    def forward(self, x):  # x(b,c,w,h) -> y(b, 4c, w/2, h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)) 
        # 图片被分为4块。x[..., ::2, ::2]即行按2叠加取，列也是，对应上面原理图的“1”块块）， x[..., 1::2, ::2]对应“3”块块，x[..., ::2, 1::2]指“2”块块，x[..., 1::2, 1::2]指“4”块块。都是每隔一个采样（采奇数列）。用cat连接这些采样图，生成通道数为12的特征图
```

# 02. Yolov5的一些相关细节

- Letterbox的操作及其应用，训练时没有采用缩减黑边的方式，还是采用传统填充的方式，即缩放到416*416大小。只是在测试，使用模型推理时，才采用缩减黑边的方式，提高目标检测，推理的速度。第一步计算长和宽的最小缩放比，第二步计算缩放后的尺寸，第三步计算最小填充；实际中使用了一次cv2.resize，一次cv2.copyMakeBorder；


# 03. Yolov5与Yolov4相比neck部分有什么不同

- Yolov5的Neck和Yolov4中一样，都采用FPN+PAN的结构，但是在Yolov4的Neck结构中，采用的都是普通的卷积操作。而Yolov5的Neck结构中，采用借鉴CSPnet设计的CSP2结构，加强网络特征融合的能力。

# 04. YOLO9000为什么可以检测9000个类？

- 采用了一种联合训练的方法，

- WordTree如何表达对象的类别？首先在训练集的构建方面：按照各个类别之间的从属关系建立一种树型结构WordTree，对于物体的标签，采用one-hot编码的形式，数据集中的每个物体的类别标签被组织成1个长度为9418的向量，向量中除了在WordTree中从该物体对应的名词到根节点的路径上出现的词对应的类别标号处为1，其余位置为0。

- 在训练的过程中，当网络遇到来自检测数据集的图片时，用完整的YOLOv2loss进行反向传播计算，当网络遇到来自分类数据集的图片时，只用分类部分的loss进行反向传播。在类别概率预测上使用层次softmax处理，是每个层次类别上分别使用softmax。

- 预测时如何确定一个WordTree所对应的对象？既然各节点预测的是条件概率，那么一个节点的绝对概率就是它到根节点路径上所有条件概率的乘积。从根节点开始向下遍历，对每一个节点，在它的所有子节点中，选择概率最大的那个（一个节点下面的所有子节点是互斥的），一直向下遍历直到某个节点的子节点概率低于设定的阈值（意味着很难确定它的下一层对象到底是哪个），或达到叶子节点，那么该节点就是该WordTree对应的对象。

- [层次化Softmax的pytorch参考链接](https://geek-docs.com/pytorch/pytorch-questions/241_pytorch_tensorflow_hierarchical_softmax_implementation.html)

- [详细参考链接](https://zhuanlan.zhihu.com/p/47575929)


# 05. FCOS如何解决重叠样本，以及centerness的作用

- 第一个问题是FCOS预测什么，其实同anchor的预测一样，预测一个4D的向量（l,r,b,t）和类别；
- 如何确认正负样本，如果一个location(x, y)落到了任何一个GT box中，那么它就为正样本，这样相对于anchor系列的正样本就会多很多；
- 如何处理一个location(x, y)对应多个GT，分两步：首先利用FPN进行多尺度预测，不太层负责不同大小的目标；然后再取最小的那个当做回归目标；
- 由于我们把中心点的区域扩大到整个物体的边框，经过模型优化后可能有很多中心离GT box中心很远的预测框，为了增加更强的约束。当loss越小时，centerness就越接近1，也就是说回归框的中心越接近真实框；
- FCOS在处理遮挡和尺度变化问题上具有优势；

# 06. Centernet为什么可以去除NMS，以及正负样本的定义

- 采用下采样代替NMS，检测当前热点的值是否比周围的八个近邻点都大，然后取100个这样的点，采用的方式是一个3x3的MaxPool，类似于anchor-based检测中nms的效果；
- 模型输出是什么？模型的最后都是加了三个网络构造来输出预测值，默认是80个类、2个预测的中心点坐标、2个中心点的偏置，对应offset，scale，hm三个头；
- 如何构建正负样本？首先根据GT求中心点坐标，其次除以下采样倍率，然后用一个高斯核来将关键点分布到特征图上，中心位置是1，其余部位逐渐降低，其中方差是一个与目标大小(也就是w和h)相关的标准差。如果某一个类的两个高斯分布发生了重叠，直接取元素间最大的就可以。在CenterNet中，每个中心点对应一个目标的位置，不需要进行overlap的判断。那么怎么去减少negative center pointer的比例呢？CenterNet是采用Focal Loss的思想。
- 重点看一下中心点预测的损失函数。当预测为1时候，[参考链接](https://zhuanlan.zhihu.com/p/66048276)

![Alt](/interview/cv-detection/centernet.jpg#pic_center=300x200)


# 07. DETR的检测算法的创新点

- DETR将目标检测看作一种set prediction问题，CNN提取基础特征，送入*Transformer做关系建模*，得到的输出通过*二分图匹配算法*与图片上的ground truth做匹配。

- 问题1：将CNN backbone输出的feature map转化为能够被Transformer Encoder处理的序列化数据的过程？维度压缩，序列化特征，结合位置编码；

- 问题2：object queries是什么？object queries是N个learnable embedding，训练刚开始时可以随机初始化。在训练过程中，因为需要生成不同的boxes，object queries会被迫使变得不同来反映位置信息，所以也可以称为leant positional encoding；

- 问题3：Loss如何设计？二分图匹配需要一一配对，前面输出的N个；只要定义好每对prediction box和image object匹配的cost；如果预测的prediction box类别和image object类别相同的概率越大（越小），或者两者的box差距越小（越大），配对的cost 越小（越大）。

- 怎么理解Query，在 DETR 中，Object query 是由 transformer 解码器产生的，它由一组预定义的向量组成，每个向量代表一个预测框。这些向量可以被视为检测模型的输出类别和空间信息的结合，其中类别信息用于区分不同的目标，而空间信息则描述了目标在图像中的位置。而 DETR 中使用 object query 向量代替 anchor box，可以在不依赖于预先设定 anchor box 的情况下进行目标检测，从而更好地适应不同大小和形状的目标。[参考链接1](https://zhuanlan.zhihu.com/p/267156624), [参考链接2](https://www.idea.edu.cn/news/2907.html)

![Alt](/interview/cv-detection//detr.png#pic_center=300x200)



