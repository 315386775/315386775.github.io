---
title: '视觉基准大模型：基于分割一切SAM模型的微调'
date: 2023-12-10
permalink: /posts/2023/12/foundation-model/
tags:
  - 大模型
  - SAM
  - 微调
---

Meta 的两个大型模型在当前 AI 开源界处于领先地位：LLM 的 LLaMA 和 CV 的 SAM。其中Segment Anything Model 是一个基于 Transformer（ViT 主干）的视觉分割基础模型。它可以通过零样本概括自动分割任何图像。另外模型还融入了Prompt：它们可以是要分割的区域上的点、要分割的对象周围的边界框或有关应分割的内容的文本提示。该模型由 3 个组件组成：图像编码器、提示编码器和掩模解码器。

![Alt](/posts/blog1210/01.png#pic_center=600x400)

但目前从实际的应用来看，其针对特定的任务表现并不是特别理想，因此微调成为一个潜在的解决方案。


## 微调大模型的基本准则及示例

## 基于预训练模型的微调示例

- 训练 SAM 需要 256 个 A100 GPU 3~5 天。每次从头开始训练整个模型的成本非常高。从预训练的基础模型进行训练或微调，这也是迁移学习的参考基线。

    **优点**：在图像编码器上我们可以采用未标记的数据（即以无监督的方式）对模型进行微调。

    **缺点**：因为未标记数据，因此也缺失任务指向性，效果可能不佳。

- SAM 提供 sam_model_registry 方法来加载其模型检查点，详细代码参考：第六部分的附录[4]。

```python
# Loading the model based on checkpoint
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
predictor = SamPredictor(sam)
# Train stage
optimizer.zero_grad()
outputs = predictor.sam(inputs)
loss = criterion(outputs, inputs)
loss.backward()
optimizer.step()
```

### 使用 Mask 标注进行特定领域的微调

- 为了解决上述第一种方案的缺点，我们需要增加标注信息，比如带有标签数据的分割任务，我们需要做一些调整：

    数据集应返回图像及其相应的分割Mask；

    损失函数应该适合分割任务；

- 详细代码参考：第六部分的附录[4]。

### SAM 特定微调

- 上述两种方式都是完全微调整个模型，存在GPU显存不足的情况，SAM 的图像编码器具有复杂的架构和许多参数。为了微调模型，我们有必要关注掩模解码器，因为它是轻量级的，因此微调起来更容易、更快且内存效率更高。

- 其中主要的问题：调用SAM的解码器```SamPredictor.predict```的问题。具有``` @torch.no_grad()```装饰器，阻止梯度的回传，因此需要重构Predict。

```python
# Image encoding
with torch.no_grad():
    image_embedding = sam_model.image_encoder(input_image)

# Prompt encoding
with torch.no_grad():
    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(points=None, boxes=box_torch, masks=None)

# Mask decoding
low_res_masks, iou_predictions = sam_model.mask_decoder(
    image_embeddings=image_embedding,
    image_pe=sam_model.prompt_encoder.get_dense_pe(),
    sparse_prompt_embeddings=sparse_embeddings,
    dense_prompt_embeddings=dense_embeddings,
    multimask_output=False,
)

# Postprocessing
upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
```

- 详细代码参考：第六部分的附录[3], 微调后的模型在文本检测的数据集上效果对比。

### 采用嵌入领域知识的微调

- 参考文献：SAM Fails to Segment Anything? -- SAM-Adapter: Adapting SAM in Underperformed Scenes: Camouflage, Shadow, Medical Image Segmentation, and More

- 提出 {SAM-Adapter}，而不是微调 SAM 网络，它通过使用简单而有效的适配器将特定于域的信息或视觉提示合并到解码网络中。

- 详细代码参考：第六部分的附录[5]。

## 参考文献及代码项目

[1] 微调应用示例：https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SAM/Fine_tune_SAM_(segment_anything)_on_a_custom_dataset.ipynb

[2] 如何微调SAM的原理：https://encord.com/blog/learn-how-to-fine-tune-the-segment-anything-model-sam/

[3] 如何微调SAM的原理对应代码： https://colab.research.google.com/drive/1F6uRommb3GswcRlPZWpkAQRMVNdVH7Ww?usp=sharing#scrollTo=r0oru8hAn6q2

[4] 微调大模型的基本方式：https://pub.towardsai.net/fine-tune-meta-sam-19f7cd4331dd

[5] 添加领域知识的微调：https://github.com/tianrun-chen/SAM-Adapter-PyTorch

[6] SAM微调集合：https://github.com/luca-medeiros/lightning-sam

[7] CT：https://github.com/rekalantar/MedSegmentAnything_SAM_LungCT

[8] 微调代码： https://colab.research.google.com/drive/1F6uRommb3GswcRlPZWpkAQRMVNdVH7Ww?usp=sharing#scrollTo=r0oru8hAn6q2

[9] 医疗影像领域的SAM：https://github.com/bowang-lab/medsam

[10] 边缘计算部署：https://github.com/ChaoningZhang/MobileSAM


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