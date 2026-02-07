import os
import time

# 必须在 import transformers / huggingface_hub 之前设置，否则仍会请求 huggingface.co 导致 10054
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print("已自动设置 HF_ENDPOINT=https://hf-mirror.com（国内镜像）")

from huggingface_hub import snapshot_download
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

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
    
    # 使用 deepseek-community 新格式（与 Transformers 原生兼容，无需 trust_remote_code 处理 processor）
    # 旧版 deepseek-ai 仓库为 multi_modality 格式，image_processor_type 易报错，不推荐。
    model_configs = {
        "small": "deepseek-community/deepseek-vl-1.3b-base",
        "base": "deepseek-community/deepseek-vl-7b-base",
        "large": "deepseek-ai/deepseek-vl-67b-base",  # 暂无 community 67b，仍用官方
    }
    
    # 根据你的配置选择（base 约 7B，纯 CPU 可跑但较慢）
    model_name = model_configs["base"]
    
    print(f"选择模型: {model_name}")
    
    # 创建保存目录
    save_dir = "./deepseek_vl_model"
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # 先用 snapshot_download 整仓下载到本地（走 HF_ENDPOINT 镜像，减少 10054）
        # 再从本地加载，避免 AutoProcessor.from_pretrained(模型名) 时反复请求被重置
        print("正在下载整个模型仓库到本地（使用镜像，请耐心等待）...")
        _with_retry(
            lambda: snapshot_download(
                repo_id=model_name,
                local_dir=save_dir,
                local_dir_use_symlinks=False,
            )
        )
        print("仓库下载完成，正在从本地加载 processor 与模型...")
        processor = AutoProcessor.from_pretrained(save_dir, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            save_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            device_map="cpu",
        )
        # 已落在 save_dir，无需再 save_pretrained
        print(f"✅ 模型已就绪: {save_dir}")
        total_gb = sum(
            os.path.getsize(os.path.join(save_dir, f))
            for f in os.listdir(save_dir)
            if os.path.isfile(os.path.join(save_dir, f))
        ) / 1024 / 1024 / 1024
        print(f"总大小: {total_gb:.2f} GB")
        
    except Exception as e:
        print(f"下载失败: {e}")
        print("尝试下载较小的模型...")
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