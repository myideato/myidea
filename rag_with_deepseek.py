"""
使用 LlamaIndex 和 DeepSeek 实现 RAG 应用
加载"智能体框架研发与落地指南.md"文档并进行查询
"""

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import OpenAI
import os

# DeepSeek API 配置
# 请设置环境变量 DEEPSEEK_API_KEY，或在此处直接填写你的 API Key
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "test-key")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# 文档路径
DOCUMENT_PATH = "智能体框架研发与落地指南.md"

# 加载文档
print(f"正在加载文档: {DOCUMENT_PATH}")
documents = SimpleDirectoryReader(
    input_files=[DOCUMENT_PATH]
).load_data()
print(f"已加载 {len(documents)} 个文档块")

# 初始化 LLM - 使用 DeepSeek（DeepSeek API 兼容 OpenAI 格式）
print("正在初始化 DeepSeek LLM...")
llm = OpenAI(
    temperature=0.1,
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)

# 创建 ServiceContext
service_context = ServiceContext.from_defaults(llm=llm)

# 创建索引
print("正在创建向量索引...")
index = VectorStoreIndex.from_documents(
    documents, 
    service_context=service_context
)
print("索引创建完成")

# 创建查询引擎
query_engine = index.as_query_engine()

# 查询
query = "文档中提到了哪些关键概念？"
print(f"\n查询问题: {query}")
print("\n正在查询...")
response = query_engine.query(query)

print("\n查询结果:")
print(response)
