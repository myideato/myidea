"""
使用 LangChain 和 DeepSeek 实现简单的 RAG 应用
"""

import os

# 配置 HuggingFace 镜像源（解决网络连接问题）
# 使用国内镜像源，如果不需要可以注释掉或设置为 None
HF_MIRROR = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
if HF_MIRROR and not os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = HF_MIRROR
    print(f"使用 HuggingFace 镜像源: {HF_MIRROR}")

# 如果需要使用代理，取消下面的注释并设置代理地址
# 示例：
# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
PROXY = os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
if PROXY:
    print(f"使用代理: {PROXY}")

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# DeepSeek API 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 文档路径
DOCUMENT_PATH = "智能体框架研发与落地指南.md"

# 初始化 LLM - 使用 DeepSeek（兼容 OpenAI 格式）
print("正在初始化 DeepSeek LLM...")
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=DEEPSEEK_BASE_URL,
    temperature=0.1,
    timeout=600.0,  # 设置超时为 600 秒（10 分钟）
    max_retries=5  # 最大重试次数
)

# 初始化 Embedding 模型
print("正在初始化 Embedding 模型...")
# 使用 HuggingFace 的轻量级 embedding 模型
# 注意：如果网络连接有问题，可以：
# 1. 使用镜像源（已配置 HF_ENDPOINT）
# 2. 使用代理（设置 HTTP_PROXY/HTTPS_PROXY 环境变量）
# 3. 手动下载模型到本地缓存目录
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("Embedding 模型初始化成功")
except Exception as e:
    print(f"初始化失败: {e}")
    print("\n提示：如果遇到网络问题，可以尝试：")
    print("1. 设置代理环境变量：set HTTP_PROXY=http://your-proxy:port")
    print("2. 使用镜像源：set HF_ENDPOINT=https://hf-mirror.com")
    print("3. 手动下载模型到本地缓存")
    raise

# 加载文档
print(f"正在加载文档: {DOCUMENT_PATH}")
loader = TextLoader(DOCUMENT_PATH, encoding='utf-8')
documents = loader.load()
print(f"已加载文档，共 {len(documents)} 个文档")

# 分割文档
print("正在分割文档...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_documents(documents)
print(f"文档已分割为 {len(texts)} 个文本块")

# 创建向量存储
print("正在生成文档向量并创建向量存储...")
vectorstore = FAISS.from_documents(texts, embeddings)
print("向量存储创建完成")

# 创建检索器
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # 返回最相关的 3 个文档块
)

# 创建自定义提示模板
prompt_template = """基于以下上下文信息回答问题。如果你不知道答案，就说不知道，不要编造答案。

上下文信息：
{context}

问题：{question}

答案："""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# 定义格式化文档的函数
def format_docs(docs):
    """将检索到的文档格式化为字符串"""
    return "\n\n".join(doc.page_content for doc in docs)

# 创建检索问答链（使用新的 LangChain API）
print("正在创建检索问答链...")
qa_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | PROMPT
    | llm
    | StrOutputParser()
)

# 查询
query = "文档中提到了哪些关键概念？"
print(f"\n查询问题: {query}")
print("\n正在查询...")

# 执行查询
result = qa_chain.invoke(query)

# 获取源文档
source_docs = retriever.invoke(query)

print("\n查询结果:")
print(result)

print("\n来源文档块:")
for i, doc in enumerate(source_docs, 1):
    print(f"\n文档块 {i}:")
    print(f"{doc.page_content[:200]}...")
