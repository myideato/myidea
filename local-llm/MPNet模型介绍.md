# MPNet 模型介绍

## 一、什么是 MPNet

**MPNet**（Masked and Permuted Pre-training for Language Understanding）是微软亚洲研究院在 2020 年提出的**预训练语言模型**，发表于 NeurIPS 2020。它结合了 BERT 和 XLNet 的优势，同时规避了两者的局限性，在多项自然语言理解任务上取得了当时的最优性能。

### 核心思想

- **融合 MLM 与 PLM**：将 BERT 的掩码语言建模（MLM）与 XLNet 的排列语言建模（PLM）相结合。
- **双向上下文 + 依赖建模**：既利用双向上下文信息，又捕捉被预测 token 之间的依赖关系。
- **位置信息一致**：通过辅助位置信息输入，减少预训练与微调阶段的位置差异。

### 主要特点

1. **性能优越**：在 GLUE、SQuAD、RACE、IMDB 等下游任务上超越 BERT、XLNet 和 RoBERTa。
2. **通用性强**：可用于文本分类、NER、问答、语义检索等多种 NLP 任务。
3. **生产就绪**：`all-mpnet-base-v2` 等变体在语义嵌入任务上表现稳定，适合实际部署。

### 典型应用

- **语义向量生成**：将句子/段落映射为 768 维稠密向量，用于语义检索、RAG、聚类。
- **文本分类**：通过序列分类微调，用于情感分析、主题分类等。
- **命名实体识别（NER）**：通过 Token 分类微调。
- **问答任务**：如 SQuAD 式阅读理解。
- **多项选择**：阅读理解、常识推理等。

---

## 二、核心原理与架构

### 2.1 BERT 与 XLNet 的局限性

| 模型 | 预训练方式 | 优点 | 局限性 |
|------|-----------|------|--------|
| **BERT** | 掩码语言建模（MLM） | 利用双向上下文，保留完整位置信息 | 假设被掩码 token 相互独立，忽略它们之间的依赖关系 |
| **XLNet** | 排列语言建模（PLM） | 捕捉被预测 token 之间的依赖关系 | 自回归预测时只能看到排列序列中前面的 token，无法看到全序列，预训练与微调存在位置信息不一致 |

### 2.2 MPNet 的改进方案

MPNet 通过以下方式统一并改进 MLM 与 PLM：

1. **引入排列语言建模（PLM）**  
   通过排列序列并自回归预测，捕捉被预测 token 之间的依赖关系，弥补 BERT 的不足。

2. **加入辅助位置信息**  
   将完整句子的位置信息作为辅助输入，让模型在预训练时也能看到全序列，减少与微调阶段的位置差异，解决 XLNet 的问题。

3. **统一视角**  
   MLM 和 PLM 可以统一理解为：将序列中的 token 分为「非预测部分」和「预测部分」两类，MPNet 在此基础上进行融合与优化。

### 2.3 模型架构

MPNet 采用与 BERT-base 相同的 Transformer 编码器结构：

| 配置项 | 规格 |
|--------|------|
| 层数 | 12 层 Transformer |
| 隐层大小 | 768 |
| 注意力头数 | 12 |
| 词汇表大小 | 30,527 |
| 总参数量 | 约 109–110M（1.09–1.1 亿） |

### 2.4 预训练数据

- 预训练语料规模：**超过 160GB** 文本数据。
- 预训练方法：掩码 + 排列语言建模（Masked and Permuted Language Modeling）。

---

## 三、与 BERT、XLNet 的对比

### 3.1 预训练方式对比

```
BERT (MLM):  [M] 我 [M] 北京 [M] 去
             → 每个 [M] 独立预测，忽略 [M] 之间的依赖

XLNet (PLM): 排列序列后自回归预测
             → 能捕捉依赖，但看不到全序列位置信息

MPNet:       结合两者 + 辅助位置信息
             → 既有依赖建模，又有完整位置信息
```

### 3.2 MPNet 的优势总结

1. **保留依赖关系**：像 XLNet 一样考虑预测 token 之间的依赖。
2. **保留位置信息**：像 BERT 一样提供完整序列的位置信息。
3. **性能更优**：在 GLUE、SQuAD 等 benchmark 上超越 BERT 和 XLNet。

---

## 四、下游任务支持

MPNet 支持多种 NLP 下游任务，在 Hugging Face Transformers 中有对应实现：

| 任务类型 | 模型类 | 说明 |
|----------|--------|------|
| 掩码语言建模 | `MPNetForMaskedLM` | 填空、完形填空 |
| 序列分类 | `MPNetForSequenceClassification` | 文本分类、情感分析 |
| Token 分类 | `MPNetForTokenClassification` | NER、词性标注 |
| 问答 | `MPNetForQuestionAnswering` | SQuAD 式阅读理解 |
| 多项选择 | `MPNetForMultipleChoice` | 阅读理解、常识推理 |
| 语义嵌入 | Sentence-Transformers 封装 | 句子向量、语义检索 |

---

## 五、语义向量生成与使用

**MPNet 可将文本映射为固定维度的语义向量（embedding）**，常用于语义检索、相似度计算、聚类等。

### 5.1 工作原理

- 使用 **MPNet 编码器** 提取 token 表示。
- 通过 **Mean Pooling** 等池化方式得到句子级向量。
- 输出 **768 维** 稠密向量，可做余弦相似度等计算。

### 5.2 使用 Sentence-Transformers（推荐）

`all-mpnet-base-v2` 是专门为语义嵌入微调的 MPNet 变体，在超过 10 亿训练对上训练，质量高、通用性强。

```python
from sentence_transformers import SentenceTransformer

# 加载模型
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# 生成语义向量
sentences = ["这是第一个句子", "这是第二个句子"]
embeddings = model.encode(sentences)  # shape: [2, 768]

# 计算相似度
from sentence_transformers import util
similarity = util.cos_sim(embeddings[0], embeddings[1])
```

### 5.3 使用 Hugging Face Transformers

若需更细粒度控制，可直接使用 Transformers，并手动实现 Mean Pooling 与归一化：

```python
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

sentences = ["这是第一个句子", "这是第二个句子"]
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

with torch.no_grad():
    model_output = model(**encoded_input)

# Mean Pooling：对非 padding token 取平均
token_embeddings = model_output.last_hidden_state
attention_mask = encoded_input['attention_mask']
input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
sentence_embeddings = sum_embeddings / sum_mask

# L2 归一化
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
```

### 5.4 常见用途

- **语义检索**：将文档转为向量，用查询向量做相似度检索。
- **RAG**：为知识库生成向量，检索与问题最相关的片段。
- **文本聚类**：对大量文本做向量化后聚类。
- **去重与相似度**：判断两段文本是否语义相近。

---

## 六、开源与使用方式

### 6.1 开源信息

- **论文**：*MPNet: Masked and Permuted Pre-training for Language Understanding* (NeurIPS 2020)
- **arXiv**：https://arxiv.org/abs/2004.09297
- **GitHub**：https://github.com/microsoft/MPNet
- **Hugging Face**：`microsoft/mpnet-base`、`sentence-transformers/all-mpnet-base-v2` 等

### 6.2 安装与加载

```bash
# 使用 Sentence-Transformers（语义嵌入场景推荐）
pip install -U sentence-transformers

# 使用 Transformers（通用 NLP 任务）
pip install transformers torch
```

### 6.3 常用模型变体

| 模型名称 | 用途 | 特点 |
|----------|------|------|
| `microsoft/mpnet-base` | 通用预训练 | 原始 MPNet，需针对任务微调 |
| `sentence-transformers/all-mpnet-base-v2` | 语义嵌入 | 已微调，质量高，适合检索、相似度 |
| `all-MiniLM-L6-v2` | 语义嵌入（轻量） | 比 MPNet 快约 5 倍，质量略低 |

### 6.4 许可

- 模型使用 Apache 2.0 等常见开源许可，具体以各模型页面为准。
- 商用前建议查看对应 Hugging Face 模型页面的 LICENSE。

---

## 七、总结

| 维度 | 说明 |
|------|------|
| **定位** | 预训练语言模型，融合 BERT 与 XLNet 优势 |
| **核心创新** | 掩码 + 排列语言建模，辅助位置信息 |
| **主要能力** | 语义向量、文本分类、NER、问答、多项选择等 |
| **典型用法** | 语义检索、RAG、聚类、相似度计算 |
| **推荐变体** | `all-mpnet-base-v2`（语义嵌入场景） |
| **适用场景** | 文本理解、语义搜索、知识库检索等 |

MPNet 是**纯文本模型**，仅处理文本输入，不涉及图像或其他模态。若需图文联合表示，应使用 CLIP 等多模态模型。
