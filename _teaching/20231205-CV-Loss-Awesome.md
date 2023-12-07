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

# 01. 人脸识别任务中，ArcFace为什么比CosFace效果好

- 首先，ArcFace 和 CosFace 都是用于人脸识别的损失函数，它们的目标都是增强类间差异，减小类内差异，从而提高人脸识别的准确性。
- 其次，ArcFace是直接在角度空间θ中最大化分类界限，而CosFace是在余弦空间cos(θ)中最大化分类界限。角度间隔比余弦间隔在对角度的影响更加直接。

![Alt](/interview/cv-detection/arcface.png#pic_center=600x400)

- ArcFace对特征向量归一化和加性角度间隔，提高了类间可分性同时加强类内紧度和类间差异，Pytorch实现如下：

```python
class ArcFace(nn.Module):
    def __init__(self, cin, cout, s=8, m=0.5):
        super().__init__()
        self.s = s
        self.sin_m = torch.sin(torch.tensor(m))
        self.cos_m = torch.cos(torch.tensor(m))
        self.cout = cout
        self.fc = nn.Linear(cin, cout, bias=False)

    def forward(self, x, label=None):
        # 计算权重向量的L2范数
        w_L2 = linalg.norm(self.fc.weight.detach(), dim=1, keepdim=True).T
        # 计算输入特征向量的L2范数
        x_L2 = linalg.norm(x, dim=1, keepdim=True)
        # 计算余弦相似度
        cos = self.fc(x) / (x_L2 * w_L2)

        if label is not None:
            sin_m, cos_m = self.sin_m, self.cos_m
            # 对标签进行one-hot编码
            one_hot = F.one_hot(label, num_classes=self.cout)
            # 计算sin和cos
            sin = (1 - cos ** 2) ** 0.5
            # 计算角度和
            angle_sum = cos * cos_m - sin * sin_m
            # 根据标签应用角度间隔
            cos = angle_sum * one_hot + cos * (1 - one_hot)
            # 缩放特征向量
            cos = cos * self.s
                        
        return cos
```

# 02. 数据增强Mixup及其变体

- 在图像分类任务中，Mixup的核心思想是以某个比例将一张图像与另外一张图像进行线性混合，同时，以相同的比例来混合这两张图像的one-hot标签。
- y是one-hot标签，⽐如yi的标签为[0,0,1]，yj的标签为[1,0,0]，此时lambda为0.2，那么此时的标签就变为0.2*[0,0,1] + 0.8*[1,0,0] = [0.8,0,0.2]，其实Mixup的⽴意很简单，就是通过这种混合的模型来增强模型的泛化性；
- 在目标检测任务中，作者对Mixup进行了修改，在图像混合过程中，为了避免图像变形，保留了图像的几何形状，并将两个图像的标签合并为一个新的数组。
- 对两幅全局图像进行像素加权组合得到增强图像。下面的Mixup变体可以分为：全局图像混合，如：ManifoldMixup和Un-Mix；区域图像混合，如：CutMix、Puzzle-Mix、Attentive-CutMix和Saliency-Mix；

```python
def mixup(imgs, labels, sampler=np.random.beta, sampler_args=(1.5, 1.5)):
    """
    Mixup two images and their labels.

    Arguments:
        imgs (list or tuple) -- list of image array which to be mixed
        labels (list or tuple) -- list of labels corresponding to images
        sampler -- ratio sampler, default is beta distribution
        sampler_args (tuple) -- parameters of sampler, default is (1.5, 1.5)

    Returns:
        mix_img (3d numpy array) -- image after mixup
        mix_label (2d numpy array) -- label after mixup
    """
    assert len(imgs) == len(labels) == 2
    img1, img2 = imgs
    label1, label2 = labels
    
    # get ratio from sampler
    ratio = max(0, min(1, sampler(*sampler_args)))
    
    # mixup two images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    w = max(w1, w2)
    mix_img = np.zeros((h, w, 3))
    mix_img[:h1, :w1] = img1 * ratio
    mix_img[:h2, :w2] += img2 * (1 - ratio)
    mix_img = mix_img.astype('uint8')
    
    # mixup two labels
    mix_label = np.vstack((label1, label2))
    
    return mix_img, mix_label
```

参考链接：https://zhuanlan.zhihu.com/p/141878389