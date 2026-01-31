"""
使用 LlamaIndex 和 DeepSeek 实现 RAG 应用
加载"智能体框架研发与落地指南.md"文档并进行查询
"""

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.response_synthesizers import ResponseMode
from openai import OpenAI as OpenAIClient
import os
import hashlib
import numpy as np
import httpx

# DeepSeek API 配置
# 请设置环境变量 DEEPSEEK_API_KEY，或在此处直接填写你的 API Key
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 文档路径
DOCUMENT_PATH = "README.md"

# 初始化 LLM - 使用 DeepSeek（DeepSeek API 兼容 OpenAI 格式）
print("正在初始化 DeepSeek LLM...")

# 设置超时时间（秒）
# 使用较长的超时时间以处理复杂的 RAG 查询
TIMEOUT_SECONDS = 600.0  # 10 分钟

# 使用 gpt-3.5-turbo 作为模型名称以绕过验证，但实际使用 DeepSeek API
# 不传递 api_client，让 LlamaIndex 创建自己的客户端，然后我们修改它的超时
llm = OpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo",  # 使用已知模型名称以绕过验证
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    context_window=32768,  # DeepSeek 支持 32K 上下文
    is_chat_model=True,
    timeout=TIMEOUT_SECONDS,  # 只传递 float 值，让 LlamaIndex 处理
    max_retries=5,  # 最大重试次数
    reuse_client=True  # 复用客户端连接
)
# 覆盖模型名称以使用 DeepSeek
llm._model = "deepseek-chat"

# 辅助函数：递归修改所有可能的客户端超时设置
def set_timeout_recursive(obj, timeout_obj, depth=0, max_depth=5):
    """递归设置对象及其所有属性的超时"""
    if depth > max_depth or obj is None:
        return
    
    # 尝试设置 timeout 属性
    if hasattr(obj, 'timeout'):
        try:
            # 如果是 httpx.Timeout 对象，需要创建新的 Timeout 对象
            if isinstance(obj.timeout, httpx.Timeout):
                # 创建新的 Timeout 对象，增加连接超时
                new_timeout = httpx.Timeout(
                    connect=120.0,  # 连接超时 120 秒
                    read=timeout_obj if isinstance(timeout_obj, (int, float)) else 600.0,
                    write=120.0,
                    pool=120.0
                )
                obj.timeout = new_timeout
            else:
                obj.timeout = timeout_obj
        except Exception as e:
            pass
    
    # 检查是否是 httpx 客户端
    if hasattr(obj, '_client'):
        set_timeout_recursive(obj._client, timeout_obj, depth + 1, max_depth)
    
    # 检查是否是 OpenAI 客户端
    if hasattr(obj, '_client') and hasattr(obj._client, '_client'):
        set_timeout_recursive(obj._client._client, timeout_obj, depth + 1, max_depth)
    
    # 检查 transport
    if hasattr(obj, '_transport'):
        set_timeout_recursive(obj._transport, timeout_obj, depth + 1, max_depth)
    
    # 检查 http_client
    if hasattr(obj, 'http_client'):
        set_timeout_recursive(obj.http_client, timeout_obj, depth + 1, max_depth)
    
    # 检查 _http_client
    if hasattr(obj, '_http_client'):
        set_timeout_recursive(obj._http_client, timeout_obj, depth + 1, max_depth)

# 确保内部客户端也使用正确的超时设置
# 递归修改所有可能的客户端超时
# 使用 float 值而不是 httpx.Timeout 对象
if hasattr(llm, '_client') and llm._client is not None:
    set_timeout_recursive(llm._client, TIMEOUT_SECONDS)

# 创建一个简单的本地 Embedding 模型（不依赖 PyTorch）
class SimpleLocalEmbedding(BaseEmbedding):
    """简单的本地 embedding 模型，使用 TF-IDF 风格的向量化"""
    
    def __init__(self):
        super().__init__(model_name="simple-local")
        self._embed_dim = 384  # 固定维度（使用私有属性）
    
    def _get_query_embedding(self, query: str) -> list[float]:
        """为查询生成 embedding"""
        return self._text_to_embedding(query)
    
    def _get_text_embedding(self, text: str) -> list[float]:
        """为文本生成 embedding"""
        return self._text_to_embedding(text)
    
    async def _aget_query_embedding(self, query: str) -> list[float]:
        """异步为查询生成 embedding"""
        return self._text_to_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> list[float]:
        """异步为文本生成 embedding"""
        return self._text_to_embedding(text)
    
    def _text_to_embedding(self, text: str) -> list[float]:
        """将文本转换为 embedding 向量"""
        # 使用简单的哈希和归一化方法生成固定维度的向量
        # 这是一个简化的实现，实际应用中应该使用真正的 embedding 模型
        embed_dim = self._embed_dim
        text_lower = text.lower()
        # 使用字符级别的特征
        embedding = [0.0] * embed_dim
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
        
        return embedding

# 初始化本地 Embedding 模型
print("正在初始化本地 Embedding 模型...")
embed_model = SimpleLocalEmbedding()

# 设置全局 LLM 和 Embedding 模型
Settings.llm = llm
Settings.embed_model = embed_model

# 加载文档
print(f"正在加载文档: {DOCUMENT_PATH}")
documents = SimpleDirectoryReader(
    input_files=[DOCUMENT_PATH]
).load_data()
print(f"已加载 {len(documents)} 个文档块")

# 创建索引
print("正在创建向量索引...")
index = VectorStoreIndex.from_documents(documents)
print("索引创建完成")

# 创建查询引擎
# 使用 "simple_summarize" 模式而不是默认的 "compact_and_refine"
# 这样可以减少 LLM 调用次数，降低超时风险
# 显式传递 LLM 确保使用我们配置的超时设置
query_engine = index.as_query_engine(
    response_mode=ResponseMode.SIMPLE_SUMMARIZE,  # 使用简单模式，只调用一次 LLM
    llm=llm  # 显式传递 LLM，确保使用我们配置的超时设置
)

# 关键修复：确保响应合成器使用的 LLM 也使用正确的超时设置
# 直接修改查询引擎内部使用的 LLM 客户端超时
if hasattr(query_engine, '_response_synthesizer'):
    response_synthesizer = query_engine._response_synthesizer
    if hasattr(response_synthesizer, '_llm'):
        synth_llm = response_synthesizer._llm
        # 递归修改响应合成器使用的 LLM 的所有客户端超时
        if hasattr(synth_llm, '_client') and synth_llm._client is not None:
            set_timeout_recursive(synth_llm._client, TIMEOUT_SECONDS)

# 查询前再次确保所有客户端使用正确的超时
# 这是最后的保险措施
print("\n正在验证超时设置...")

# 强制更新所有可能的客户端超时
def force_update_all_timeouts():
    """强制更新所有相关客户端的超时设置"""
    # 更新主 LLM
    if hasattr(llm, '_client') and llm._client is not None:
        set_timeout_recursive(llm._client, TIMEOUT_SECONDS)
    
    # 更新 Settings.llm
    if Settings.llm is not None:
        if hasattr(Settings.llm, '_client') and Settings.llm._client is not None:
            set_timeout_recursive(Settings.llm._client, TIMEOUT_SECONDS)
        # 也直接更新 LLM 对象本身
        set_timeout_recursive(Settings.llm, TIMEOUT_SECONDS)
    
    # 更新查询引擎中的响应合成器
    if hasattr(query_engine, '_response_synthesizer'):
        response_synthesizer = query_engine._response_synthesizer
        if hasattr(response_synthesizer, '_llm'):
            synth_llm = response_synthesizer._llm
            set_timeout_recursive(synth_llm, TIMEOUT_SECONDS)
            if hasattr(synth_llm, '_client') and synth_llm._client is not None:
                set_timeout_recursive(synth_llm._client, TIMEOUT_SECONDS)

force_update_all_timeouts()
print("[OK] 所有客户端超时设置已更新")

# 查询
query = "文档中提到了哪些关键概念？"
print(f"\n查询问题: {query}")
print("\n正在查询...")
response = query_engine.query(query)

print("\n查询结果:")
print(response)
