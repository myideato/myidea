import os
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor
import torch

# 国内网络访问 Hugging Face 常被重置，使用镜像可解决 WinError 10054
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print("已自动设置 HF_ENDPOINT=https://hf-mirror.com（国内镜像）")

def _with_retry(fn, max_retries=5, base_delay=3):
    """对易断连的请求做重试（指数退避）"""
    last_err = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"请求失败，{delay}s 后重试 ({attempt + 1}/{max_retries}): {e}")
                time.sleep(delay)
    raise last_err

def download_deepseek_vl():
    """下载DeepSeek-VL模型"""
    print("正在下载DeepSeek-VL模型...")
    
    # 模型配置（须使用 HF 上的完整 repo 名：-base 或 -chat）
    model_configs = {
        "small": "deepseek-ai/deepseek-vl-1.3b-base",
        "base": "deepseek-ai/deepseek-vl-7b-base",
        "large": "deepseek-ai/deepseek-vl-67b-base"  # 需要大量GPU显存
    }
    
    # 根据你的配置选择基础版（纯CPU可运行，但推理较慢）
    model_name = model_configs["base"]
    
    print(f"选择模型: {model_name}")
    
    # 创建保存目录
    save_dir = "./deepseek_vl_model"
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # 下载tokenizer（带重试，应对连接被重置）
        print("下载tokenizer...")
        tokenizer = _with_retry(
            lambda: AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        )
        tokenizer.save_pretrained(save_dir)
        
        # 下载图像处理器
        print("下载图像处理器...")
        processor = _with_retry(
            lambda: AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        )
        
        # 下载模型（使用低精度减少内存，带重试）
        print("下载模型（这可能需要一些时间）...")
        model = _with_retry(
            lambda: AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                device_map="cpu"  # 强制使用CPU
            )
        )
        
        # 保存模型
        model.save_pretrained(save_dir)
        
        print(f"✅ 模型已保存到: {save_dir}")
        print(f"总大小: {sum(os.path.getsize(os.path.join(save_dir, f)) for f in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, f))) / 1024 / 1024 / 1024:.2f} GB")
        
    except Exception as e:
        print(f"下载失败: {e}")
        print("尝试下载较小的模型...")
        # 备选方案：使用较小的模型
        download_lightweight_model()

def download_lightweight_model():
    """下载轻量级替代模型"""
    print("下载轻量级模型: BLIP-2 (更适合CPU推理)")
    
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    
    model_name = "Salesforce/blip2-opt-2.7b"
    save_dir = "./blip2_model"
    
    os.makedirs(save_dir, exist_ok=True)
    
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    processor.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
    
    print(f"✅ 轻量模型已保存: {save_dir}")

if __name__ == "__main__":
    download_deepseek_vl()