"""
使用 OpenAI SDK 和 DeepSeek 实现简单的 RAG 应用
不依赖 LlamaIndex 或 LangChain，直接使用 OpenAI SDK
"""

from openai import OpenAI
import os
import hashlib
import numpy as np

# DeepSeek API 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 文档路径
DOCUMENT_PATH = "README.md"

# 初始化 OpenAI 客户端（用于 DeepSeek）
print("正在初始化 DeepSeek 客户端...")
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    timeout=600.0,  # 设置超时为 600 秒（10 分钟）
    max_retries=5  # 最大重试次数
)

# 简单的文本分割函数
def split_text(text, chunk_size=1000, chunk_overlap=200):
    """简单的文本分割函数"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap
        if start >= len(text):
            break
    return chunks

# 简单的 embedding 函数（使用哈希）
def simple_embedding(text):
    """简单的 embedding 函数，使用哈希生成向量"""
    embed_dim = 384
    text_lower = text.lower()
    embedding = [0.0] * embed_dim
    
    # 使用字符级别的特征
    for i, char in enumerate(text_lower[:embed_dim]):
        embedding[i] = float(ord(char)) / 128.0 - 1.0
    
    # 添加文本长度特征
    text_hash = int(hashlib.md5(text.encode()).hexdigest(), 16)
    for i in range(min(32, embed_dim)):
        embedding[i] = (text_hash >> (i * 2)) % 256 / 128.0 - 1.0
    
    # 归一化
    norm = sum(x * x for x in embedding) ** 0.5
    if norm > 0:
        embedding = [x / norm for x in embedding]
    
    return np.array(embedding)

# 简单的相似度计算（余弦相似度）
def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

# 加载文档
print(f"正在加载文档: {DOCUMENT_PATH}")
with open(DOCUMENT_PATH, 'r', encoding='utf-8') as f:
    content = f.read()
print(f"已加载文档，共 {len(content)} 个字符")

# 分割文档
print("正在分割文档...")
text_chunks = split_text(content, chunk_size=1000, chunk_overlap=200)
print(f"文档已分割为 {len(text_chunks)} 个文本块")

# 为每个文本块生成 embedding
print("正在生成文档向量...")
chunk_embeddings = [simple_embedding(chunk) for chunk in text_chunks]
print("文档向量生成完成")

# 查询
query = "文档中提到了哪些关键概念？"
print(f"\n查询问题: {query}")
print("\n正在检索相关文档...")

# 为查询生成 embedding
query_embedding = simple_embedding(query)

# 计算相似度并找到最相关的文档块
similarities = [cosine_similarity(query_embedding, chunk_emb) for chunk_emb in chunk_embeddings]
top_k = 3
top_indices = np.argsort(similarities)[-top_k:][::-1]

# 获取最相关的文档块
relevant_chunks = [text_chunks[i] for i in top_indices]
print(f"找到 {len(relevant_chunks)} 个相关文档块")

# 构建上下文
context = "\n\n".join([f"文档块 {i+1}:\n{chunk}" for i, chunk in enumerate(relevant_chunks)])

# 构建提示
prompt = f"""基于以下上下文信息回答问题。如果你不知道答案，就说不知道，不要编造答案。

上下文信息：
{context}

问题：{query}

答案："""

# 调用 DeepSeek API
print("\n正在查询 DeepSeek API...")
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一个有用的助手，能够基于提供的上下文信息回答问题。"},
        {"role": "user", "content": prompt}
    ],
    temperature=0.1,
    stream=False
)

print("\n查询结果:")
print(response.choices[0].message.content)
print("\n来源文档块:")
for i, idx in enumerate(top_indices, 1):
    print(f"\n文档块 {i} (相似度: {similarities[idx]:.4f}):")
    print(f"{text_chunks[idx][:200]}...")
