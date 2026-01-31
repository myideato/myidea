"""
使用 LlamaIndex 和 DeepSeek 实现简单的 RAG 应用

依赖（在已有 llama-index、faiss-cpu、sentence-transformers、openai 基础上）：
  pip install llama-index-embeddings-huggingface llama-index-vector-stores-faiss llama-index-llms-openai-like
"""

import os

# 配置 HuggingFace 镜像源（解决网络连接问题）
HF_MIRROR = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
if HF_MIRROR and not os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = HF_MIRROR
    print(f"使用 HuggingFace 镜像源: {HF_MIRROR}")

PROXY = os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
if PROXY:
    print(f"使用代理: {PROXY}")

# LlamaIndex 核心
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter

# Embedding：LlamaIndex 原生 HuggingFace（需安装 llama-index-embeddings-huggingface）
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 向量存储：FAISS（需安装 llama-index-vector-stores-faiss, faiss-cpu）
import faiss
from llama_index.vector_stores.faiss import FaissVectorStore

# LLM：OpenAI 兼容接口，用于 DeepSeek（需安装 llama-index-llms-openai-like）
from llama_index.llms.openai_like import OpenAILike

# DeepSeek API 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 文档路径
DOCUMENT_PATH = "智能体框架研发与落地指南.md"

# 初始化 LLM - 使用 DeepSeek（OpenAI 兼容 API）
print("正在初始化 DeepSeek LLM...")
llm = OpenAILike(
    model="deepseek-chat",
    api_base=DEEPSEEK_BASE_URL,
    api_key=DEEPSEEK_API_KEY,
    is_chat_model=True,
    is_function_calling_model=False,
    context_window=128000,
    timeout=600.0,
)
Settings.llm = llm

# 初始化 Embedding 模型（与 rag_langchain 相同：all-MiniLM-L6-v2）
print("正在初始化 Embedding 模型...")
try:
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        trust_remote_code=True,
    )
    Settings.embed_model = embed_model
    print("Embedding 模型初始化成功")
except Exception as e:
    print(f"初始化失败: {e}")
    print("\n提示：若遇网络问题，可设置代理或 HF_ENDPOINT 镜像")
    raise

# 文本分块配置（与 LangChain 版本一致：chunk_size=1000, overlap=200）
node_parser = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

# 加载文档
print(f"正在加载文档: {DOCUMENT_PATH}")
reader = SimpleDirectoryReader(input_files=[DOCUMENT_PATH])
documents = reader.load_data()
print(f"已加载文档，共 {len(documents)} 个文档")

# 创建 FAISS 向量存储（all-MiniLM-L6-v2 维度为 384）
embed_dim = 384
faiss_index = faiss.IndexFlatL2(embed_dim)
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 构建索引：使用自定义分块与向量存储
print("正在生成文档向量并创建索引...")
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    transformations=[node_parser],
    show_progress=True,
)
print("索引创建完成")

# 创建查询引擎（检索 top_k=3，与 LangChain 版本一致）
query_engine = index.as_query_engine(
    similarity_top_k=3,
    response_mode="compact",  # 将检索到的上下文紧凑地用于生成
)

# 自定义系统提示，与 LangChain 版本语义一致
from llama_index.core.prompts import PromptTemplate

qa_prompt_tpl = (
    "基于以下上下文信息回答问题。如果你不知道答案，就说不知道，不要编造答案。\n\n"
    "上下文信息：\n"
    "{context_str}\n\n"
    "问题：{query_str}\n\n"
    "答案："
)
query_engine.update_prompts({"response_synthesizer:text_qa_template": PromptTemplate(qa_prompt_tpl)})

# 查询
query = "文档中提到了哪些关键概念？"
print(f"\n查询问题: {query}")
print("\n正在查询...")

response = query_engine.query(query)

print("\n查询结果:")
print(response.response)

print("\n来源文档块:")
for i, node in enumerate(response.source_nodes, 1):
    text = node.node.get_content()
    print(f"\n文档块 {i}:")
    print(f"{text[:200]}...")
