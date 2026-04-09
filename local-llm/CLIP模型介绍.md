# CLIP 模型介绍

## 一、什么是 CLIP

**CLIP**（Contrastive Language–Image Pre-training）是 OpenAI 在 2021 年提出的**视觉–语言预训练模型**，用于学习图像和文本的联合表示。

### 核心思想

- **对比学习**：在大量「图像–文本」对上训练，让匹配的图像和文本在特征空间中靠近，不匹配的远离。
- **多模态**：同时处理图像和文本，把两者映射到同一特征空间，便于跨模态检索和推理。

### 主要特点

1. **零样本能力**：训练时见过大量图文对，可以泛化到新类别，无需针对新任务微调。
2. **文本引导**：用自然语言描述任务（如“一只猫”“红色汽车”），模型根据文本检索或分类图像。
3. **灵活应用**：可用于图像分类、图文检索、图像生成（如 DALL·E）等。

### 典型应用

- **图像分类**：用文本描述类别，对图像做零样本分类。
- **图文检索**：给定文本找图像，或给定图像找文本。
- **图像生成**：作为文本–图像对齐模块，用于 DALL·E 等模型。
- **下游任务**：作为视觉编码器，用于目标检测、分割等。

### 技术要点

- 使用 **ResNet** 或 **ViT** 作为图像编码器。
- 使用 **Transformer** 作为文本编码器。
- 通过对比损失（如 InfoNCE）训练，使匹配的图文对相似度高、不匹配的相似度低。

---

## 二、图像与文本向量生成

**CLIP 可以对图片和文本分别生成向量（embedding）**，且这些向量在同一个特征空间里，可以直接比较。

### 工作原理

CLIP 有两个编码器：

1. **图像编码器**：输入图像 → 输出图像向量（如 512 维或 768 维）
2. **文本编码器**：输入文本 → 输出文本向量（相同维度）

两者输出在同一向量空间，因此可以计算相似度（如余弦相似度）做图文匹配。

### 使用示例（Python）

```python
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 加载模型
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 图像 → 向量
image = Image.open("your_image.jpg")
inputs = processor(images=image, return_tensors="pt")
image_embeddings = model.get_image_features(**inputs)  # shape: [1, 512]

# 文本 → 向量
text_inputs = processor(text=["a photo of a cat", "a photo of a dog"], return_tensors="pt", padding=True)
text_embeddings = model.get_text_features(**text_inputs)  # shape: [2, 512]

# 计算相似度（余弦相似度）
similarity = (image_embeddings @ text_embeddings.T).softmax(dim=-1)
```

### 常见用途

- **图文检索**：用文本向量在图像向量库中检索
- **零样本分类**：用类别文本的向量与图像向量比较，选最相似的类别
- **语义搜索**：把图像和文本都转成向量，做跨模态搜索

---

## 三、开源情况

**CLIP 是开源模型。**

### 开源信息

- **论文**：2021 年 OpenAI 发布 *Learning Transferable Visual Models From Natural Language Supervision*
- **代码**：GitHub 开源 - https://github.com/openai/CLIP
- **模型权重**：官方提供多种预训练权重（如 ResNet-50、ViT-B/32、ViT-L/14 等）

### 使用方式

1. **官方仓库**：`pip install git+https://github.com/openai/CLIP.git`
2. **Hugging Face**：`transformers` 和 `open_clip` 都支持 CLIP
   - `openai/clip-vit-base-patch32`
   - `openai/clip-vit-large-patch14`
   - 等
3. **OpenCLIP**：社区维护的扩展版，支持更多预训练模型和数据集
   - https://github.com/mlfoundations/open_clip

### 许可

- 使用 MIT 许可证，可商用
- 部分预训练权重可能有单独说明，使用前建议查看对应仓库的 LICENSE

---

## 四、官方 API 情况

**没有官方的免费 CLIP API。**

### 官方情况

- OpenAI 只开源了 CLIP 的代码和权重，**没有提供 CLIP 的在线 API 服务**。
- OpenAI 的 API（如 GPT、DALL-E、Embeddings）都是付费的，且没有单独的 CLIP 接口。

### 可行方案

#### 1. 本地部署（免费）

在本地用开源实现跑 CLIP，不调用任何 API：

```bash
pip install transformers torch pillow
```

```python
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# 本地推理，完全免费
```

#### 2. 第三方 API（非官方）

- **Replicate**：有 CLIP 模型，按调用量计费，有免费额度
- **Hugging Face Inference API**：可部署 CLIP，有免费 tier
- 其他云服务商也可能提供类似 embedding 接口，但都不是 OpenAI 官方 CLIP API

#### 3. 自建 API

在服务器上部署 CLIP，自己封装成 HTTP API，供内部或外部调用。

### 结论

没有官方免费 CLIP API。若想免费使用，建议在本地或自建服务器上部署 CLIP 模型。
