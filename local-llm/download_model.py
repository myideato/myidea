import os
import time

# 必须在 import transformers / huggingface_hub 之前设置，否则仍会请求 huggingface.co 导致 10054
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print("已自动设置 HF_ENDPOINT=https://hf-mirror.com（国内镜像）")

from huggingface_hub import snapshot_download
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch


def _repo_root() -> str:
    """含 download_model.py 的仓库根目录（myidea），即 local-llm 的父目录。"""
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


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
    
    # 使用 deepseek-community 新格式（与 Transformers 原生兼容）
    # chat 版本经指令微调，对话/看图问答更稳定，推荐。
    model_configs = {
        "small": "deepseek-community/deepseek-vl-1.3b-chat",
        "base": "deepseek-community/deepseek-vl-7b-chat",
        "large": "deepseek-ai/deepseek-vl-67b-chat",  # 暂无 community 67b，仍用官方
    }
    
    # 根据你的配置选择（base 约 7B，纯 CPU 可跑但较慢）
    model_name = model_configs["base"]
    
    print(f"选择模型: {model_name}")
    
    # 创建保存目录
    save_dir = "./deepseek_vl_chat_model"
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
        # print("尝试下载较小的模型...")
        # download_blip2_model()

def download_blip2_model():
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

def download_qwen25_vl():
    """下载 Qwen2.5-VL-7B-Instruct 到仓库根目录下的 qwen2_5_vl_7b_instruct（原生视频理解、结构化打标）。"""
    from transformers import AutoProcessor

    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    save_dir = os.path.abspath(os.path.join(_repo_root(), "qwen2_5_vl_7b_instruct"))

    print(f"正在下载 Qwen2.5-VL: {model_name}")
    print(f"保存目录: {save_dir}")

    os.makedirs(save_dir, exist_ok=True)

    try:
        print("正在下载整个模型仓库到本地（使用镜像，请耐心等待，约 15GB+）...")
        _with_retry(
            lambda: snapshot_download(
                repo_id=model_name,
                local_dir=save_dir,
                local_dir_use_symlinks=False,
            )
        )
        print("仓库下载完成，正在从本地校验 processor（不加载完整权重以省内存）...")
        AutoProcessor.from_pretrained(save_dir, trust_remote_code=True)

        print(f"✅ Qwen2.5-VL 已就绪: {save_dir}")
        total_gb = sum(
            os.path.getsize(os.path.join(save_dir, f))
            for f in os.listdir(save_dir)
            if os.path.isfile(os.path.join(save_dir, f))
        ) / 1024 / 1024 / 1024
        print(f"总大小: {total_gb:.2f} GB")
    except Exception as e:
        print(f"下载失败: {e}")
        raise


def _print_dir_size_gb(save_dir: str) -> None:
    total = 0
    for root, _dirs, files in os.walk(save_dir):
        for name in files:
            fp = os.path.join(root, name)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    print(f"总大小: {total / (1024**3):.2f} GB")


def download_pointllm(
    model_name: str = "RunsenXu/PointLLM_7B_v1.2",
    save_dir: str | None = None,
) -> None:
    """下载 PointLLM 预训练权重（彩色点云 VLM，与官方 GitHub / HF 发布一致）。默认 7B v1.2。"""
    if save_dir is None:
        save_dir = os.path.abspath(os.path.join(_repo_root(), "pointllm_7b_v1_2"))

    print(f"正在下载 PointLLM: {model_name}")
    print(f"保存目录: {save_dir}")

    os.makedirs(save_dir, exist_ok=True)

    try:
        print("正在下载整个模型仓库到本地（使用镜像，请耐心等待）...")
        _with_retry(
            lambda: snapshot_download(
                repo_id=model_name,
                local_dir=save_dir,
                local_dir_use_symlinks=False,
            )
        )
        print(f"✅ PointLLM 已就绪: {save_dir}")
        _print_dir_size_gb(save_dir)
    except Exception as e:
        print(f"下载失败: {e}")
        raise


def download_gpt4point(
    model_name: str = "alexzyqi/GPT4Point",
    save_dir: str | None = None,
) -> None:
    """下载 GPT4Point 预训练检查点（OPT 2.7B 阶段权重等，见 Pointcept 论文 / HF alexzyqi/GPT4Point）。"""
    if save_dir is None:
        save_dir = os.path.abspath(os.path.join(_repo_root(), "gpt4point_weight"))

    print(f"正在下载 GPT4Point: {model_name}")
    print(f"保存目录: {save_dir}")

    os.makedirs(save_dir, exist_ok=True)

    try:
        print("正在下载整个模型仓库到本地（使用镜像，请耐心等待）...")
        _with_retry(
            lambda: snapshot_download(
                repo_id=model_name,
                local_dir=save_dir,
                local_dir_use_symlinks=False,
            )
        )
        print(f"✅ GPT4Point 已就绪: {save_dir}")
        _print_dir_size_gb(save_dir)

        bert_dir = os.path.join(save_dir, "bert-base-uncased")
        marker = os.path.join(bert_dir, "config.json")
        if os.path.isfile(marker):
            print(f"已存在 bert-base-uncased，跳过: {bert_dir}")
        else:
            print("正在下载 bert-base-uncased（Q-Former / 分词器，ply_point_tagger 离线加载需要）…")
            os.makedirs(bert_dir, exist_ok=True)
            _with_retry(
                lambda: snapshot_download(
                    repo_id="bert-base-uncased",
                    local_dir=bert_dir,
                    local_dir_use_symlinks=False,
                )
            )
            print(f"✅ bert-base-uncased 已就绪: {bert_dir}")

        opt_dir = os.path.join(save_dir, "opt-2.7b")
        opt_cfg = os.path.join(opt_dir, "config.json")
        if os.path.isfile(opt_cfg):
            print(f"已存在 facebook/opt-2.7b 快照，跳过: {opt_dir}")
        else:
            print("正在下载 facebook/opt-2.7b（语言模型底座，体积较大，请耐心等待）…")
            os.makedirs(opt_dir, exist_ok=True)
            _with_retry(
                lambda: snapshot_download(
                    repo_id="facebook/opt-2.7b",
                    local_dir=opt_dir,
                    local_dir_use_symlinks=False,
                )
            )
            print(f"✅ opt-2.7b 已就绪: {opt_dir}")
    except Exception as e:
        print(f"下载失败: {e}")
        raise


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="下载视觉语言模型到仓库根目录")
    p.add_argument(
        "--model",
        default="deepseek-vl",
        choices=(
            "deepseek-vl",
            "qwen25-vl",
            "blip2",
            "pointllm",
            "gpt4point",
        ),
        help=(
            "deepseek-vl（默认）: DeepSeek-VL-7b-chat；"
            "qwen25-vl: Qwen2.5-VL-7B-Instruct；"
            "blip2: Salesforce/blip2-opt-2.7b；"
            "pointllm / gpt4point: 点云多模态"
        ),
    )
    args = p.parse_args()
    if args.model == "deepseek-vl":
        download_deepseek_vl()
    elif args.model == "qwen25-vl":
        download_qwen25_vl()
    elif args.model == "blip2":
        download_blip2_model()
    elif args.model == "pointllm":
        download_pointllm()
    elif args.model == "gpt4point":
        download_gpt4point()