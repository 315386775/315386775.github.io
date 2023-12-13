---
title: '有意思的损失函数：一文详细解释Yolo中Objectness的重要性'
date: 2023-12-13
permalink: /posts/2023/12/loss-objectness/
tags:
  - Objectness
  - Yolo目标检测
  - 损失函数
---

损失函数对不同的框进行不同的处理，最佳框与所有其他框之间的区分机制是 YOLO 损失的核心。使用单独的对象置信度损失 objectness 来处理分数确实比将类概率 confidence 视为分数表现得更好，在SSD目标检测中考虑类概率作为置信度分数其效果要明显差于带置信度的Yolo模型。

![Alt](/posts/blog1213/02.jpg#pic_center=600x400)

例如，在上图中，我们期望红色框的对象性较低，蓝色框的对象性中等，绿色框的对象性高。这是因为绿色框“紧密”地贴合我们的物体，而蓝色框则松散地贴合，而红色框根本不包含任何物体。

# Yolov5损失函数的构成

最佳框单独产生边界框损失 （由于与对象不太完美匹配）和分类损失（由于分类错误），这会推动与这些框相关的网络参数以改善框的位置和分类，同时这些框也会和其他所有非最佳框一起参与置信度损失。

Yolov5的三个损失项：

- 分类损失(cls_loss)：判断模型是否能够准确地识别出图像中的对象，并将其分类到正确的类别中。

- 边界框损失(box_loss)：用于衡量模型预测的边界框与真实边界框之间的差异。

- 置信度损失(obj_loss)：模型预测边界框覆盖对象的程度。

# 我们为什么需要 objectness loss？

对于每个边界框的预测，都会有一个与之相关的预测值，称为“objectness”。Objectness loss 项教会了网络如何预测正确的IoU，而坐标损失则教会了网络如何预测更好的边界框（最终将IoU推向1.0）

```类别置信度 = 类别分数 * objectness loss``

在推理时，我们通常会对每个对象预测有多个具有不同覆盖范围的边界框。我们希望后处理算法选择以最精确方式覆盖对象的边界框。我们还希望选择能够为对象提供正确类别预测的边界框。算法如何知道选择哪个边界框？

首先，objectness 告诉我们框的覆盖度有多好，因此具有非常小objectness（<0.005）的边界框会被丢弃，甚至不会进入NMS模块。这有助于消除大约90%的边界框。

![Alt](/posts/blog1213/01.jpg#pic_center=600x400)

其次，对于每个类别NMS是单独进行的，因此类别分数会根据边界框的objectness进行缩放，以进行有意义的比较。如果我们有两个具有高重叠的边界框，第一个的objectness为0.9，人的概率为0.8（加权得分为0.72），第二个的objectness为0.5，人的概率为0.8（加权得分为0.40），那么第一个边界框将被保留，而第二个边界框将在NMS中被舍弃，因为第一个边界框的objectness使其更加可信。

# 为什么在训练过程中要区别对待“最佳边界框”？

想象一位教授有以下教学策略：在第一次作业中，她寻找表现良好的学生，并努力检查和评分他们的作业，以便他们在该科目中取得优异的成绩。为了专注，她不会费心去批改表现不佳的学生的作业。相反，她会给他们一个机会，在下一次作业中取得优异的成绩。

一方面，我们希望训练模型能够有效地收敛。网络具有丰富的参数，每个参数都有足够的工作要做，因此没有必要急于一次性优化所有参数。最好是利用一些边界框的相对成功，只推动它们成功地捕捉这种类型的对象。

另一方面，我们希望所有的边界框都经历objectness loss，因为我们希望所有的边界框都学会判断它们是好还是坏，因为后续的NMS需要这个权重。


# 可作为面试实战的问题

考虑到objectness损失的重要性，我们思考一个问题：

- 如果我们在训练中改变objectness的权重（self.hyp['obj']），我们是否仍然应该在检测中简单地将objectness分数和cls分数相乘？
  
- 如果是，如果我们设置“self.hyp[‘obj’]=0”会怎么样？通过这样做，在训练期间将不会控制客观性分数。

- 为什么objectness损失会随着图像大小而变化？其受到正样本和负样本之间极度不平衡的影响。当图像放大时，其中的对象数量保持不变，因此不平衡性增加（变得更糟）。损失增益将按比例进行补偿。

# objectness loss 源码

```python

# Objectness
# iou.detach()  不会更新iou梯度  iou并不是反向传播的参数 所以不需要反向传播梯度信息
# iou.shape = [1659]
iou = iou.detach().clamp(0).type(tobj.dtype)
# 这里对 iou 进行排序再做一个优化：当一个正样本出现多个 GT 的情况也就是同一个 grid 中有两个 gt (密集型且形状差不多物体)
# There maybe several GTs match the same anchor when calculate ComputeLoss in the scene with dense targets
if self.sort_obj_iou:
    # https://github.com/ultralytics/yolov5/issues/3605
    # There maybe several GTs match the same anchor when calculate ComputeLoss in the scene with dense targets
    j = iou.argsort()
    # 如果同一个 grid 出现两个 GT 那么经过排序之后每个 grid 中的 score_iou 都能保证是最大的
    # (小的会被覆盖 因为同一个grid坐标肯定相同)那么从时间顺序的话, 最后一个总是和最大的 iou 去计算 loss
    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
    # 预测信息有置信度 但是真实框信息是没有置信度的 所以需要我们人为的给一个标准置信度
    # self.gr是iou ratio [0, 1]  self.gr越大置信度越接近iou  self.gr越小置信度越接近1(人为加大训练难度)
    if self.gr < 1:
    iou = (1.0 - self.gr) + self.gr * iou
tobj[b, a, gj, gi] = iou  # iou ratio
```

[1] 详细内容请参阅[MarkAI Blog](https://315386775.github.io/year-archive/) 
[2] 更多资料请参阅[MarkAI Github](https://github.com/315386775?tab=repositories) 
[3] 深度解析请参阅[2024年千道算法面试题综述](https://github.com/315386775/DeepLearing-Interview-Awesome-2024) 














