"""
使用本地 Qwen2.5-VL 对视频做原生编码与结构化打标（无需抽帧落地成多张图）。
依赖: pip install transformers torch accelerate pillow
官方解码（默认）: pip install av（PyAV，torchvision 读视频需要）
可选: pip install numpy opencv-python，并使用 --video_decode opencv（不装 av 时）
下载模型: python download_model.py --model qwen25-vl（模型在仓库根目录 qwen2_5_vl_7b_instruct）
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import TextStreamer

def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def _safe_filename_segment(name: str) -> str:
    """用于默认输出文件名的片段，避免路径非法字符。"""
    s = re.sub(r'[<>:"/\\|?*]', "_", name.strip())
    s = s.strip(" .")
    return s or "untitled"


def _default_output_json_path(video_path: str, model_path: str) -> str:
    video_abs = os.path.abspath(os.path.normpath(video_path))
    model_abs = os.path.abspath(os.path.normpath(model_path))
    video_dir = os.path.dirname(video_abs)
    video_stem = os.path.splitext(os.path.basename(video_abs))[0]
    model_name = os.path.basename(model_abs.rstrip(os.sep + (os.altsep or "")))
    base = f"{_safe_filename_segment(video_stem)}_{_safe_filename_segment(model_name)}.json"
    return os.path.join(video_dir, base)


# 与 download_model.py 默认目录一致（仓库根目录 myidea/qwen2_5_vl_7b_instruct）
_DEFAULT_MODEL_DIR = os.path.join(_repo_root(), "qwen2_5_vl_7b_instruct")

VIDEO_STRUCTURED_PROMPT = """你是一个视频分析助手。请根据视频内容只输出一个 JSON 对象，不要 markdown 代码块，不要其他说明文字。
格式示例：
{"description": "", "scene": "", "objects": "", "main_actions": "", "person": "", "environment": "", "emotion": "", "color": "", "location": ""}

字段说明：
- description: 视频整体内容描述（50-200字），含时间顺序上的主要变化
- scene: 一句话概括场景与主题
- objects: 主要物体与场景元素，逗号分隔
- main_actions: 主要动作或事件，按大致时间顺序，逗号分隔
- person: 人物相关（动作、穿着等），无人或看不清填 null
- environment: 环境（室内外、天气、光线等），无法判断填 null
- emotion: 情绪或氛围
- color: 画面色彩与影调（主色/配色、冷暖、饱和度、明暗倾向等），逗号分隔；无法判断填 null
- location: 场所类型（户外/室内/办公室等）
"""


def _default_device_dtype():
    if torch.cuda.is_available():
        dev = "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dev = "cpu"
        dtype = torch.float32
    return dev, dtype


def _load_video_rgb_numpy(path: str, sample_fps: float) -> np.ndarray:
    """
    用 OpenCV 解码并按约 sample_fps 每秒取样（与 transformers 视频处理器在 path 模式下的取样方式一致），
    避免依赖 PyAV / torchvision.read_video。
    返回 uint8 RGB，形状 (T, H, W, 3)。
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError(
            "无法进行视频解码。请安装其一：pip install opencv-python  或  pip install av（PyAV）"
        ) from e

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"无法用 OpenCV 打开视频: {path}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices_set: Optional[set[int]] = None
    if total_reported > 0:
        num_frames = int(total_reported / native_fps * sample_fps)
        num_frames = max(1, min(num_frames, total_reported))
        indices = np.arange(0, total_reported, total_reported / num_frames).astype(np.int64)
        indices_set = set(indices.tolist())

    frames_bgr: List[np.ndarray] = []
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if indices_set is None:
                frames_bgr.append(frame)
            elif index in indices_set:
                frames_bgr.append(frame)
            index += 1
    finally:
        cap.release()

    if indices_set is None and frames_bgr:
        total = len(frames_bgr)
        num_frames = max(1, int(total / native_fps * sample_fps))
        num_frames = min(num_frames, total)
        pick = np.arange(0, total, total / num_frames).astype(np.int64)
        frames_bgr = [frames_bgr[int(i)] for i in pick]

    if not frames_bgr:
        raise RuntimeError(f"未读取到任何帧: {path}")

    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    return np.stack(rgb, axis=0)


class Qwen25VideoTagger:
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        max_new_tokens: int = 1024,
        attn_implementation: str = "sdpa",
    ):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self.model_path = os.path.abspath(os.path.normpath(model_path or _DEFAULT_MODEL_DIR))
        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(
                f"未找到本地模型目录: {self.model_path}\n请先运行: python local-llm/download_model.py --model qwen25-vl"
            )

        if device:
            self.device = device
            self.dtype = torch.float32 if device == "cpu" else (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16
            )
        else:
            self.device, self.dtype = _default_device_dtype()

        print(f"加载 Qwen2.5-VL: {self.model_path}")
        print(f"设备: {self.device}, dtype: {self.dtype}")

        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)

        load_kw: Dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
            "device_map": self.device if self.device == "cpu" else "auto",
            "attn_implementation": attn_implementation,
        }

        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_path, **load_kw)
        except (TypeError, ValueError):
            load_kw.pop("attn_implementation", None)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_path, **load_kw)
        self.model.eval()
        self._max_new_tokens = max_new_tokens
        self._tokenizer = getattr(self.processor, "tokenizer", None)

    def tag_video(
        self,
        video_path: str,
        prompt: Optional[str] = None,
        fps: float = 1.0,
        video_decode: str = "official",
    ) -> Dict[str, Any]:
        video_path = os.path.abspath(os.path.normpath(video_path))
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"视频不存在: {video_path}")

        text_prompt = prompt if prompt else VIDEO_STRUCTURED_PROMPT
        if video_decode == "official":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "path": video_path},
                        {"type": "text", "text": text_prompt},
                    ],
                }
            ]
            template_kw: Dict[str, Any] = {
                "add_generation_prompt": True,
                "tokenize": True,
                "return_dict": True,
                "return_tensors": "pt",
                "fps": float(fps),
            }
        elif video_decode == "opencv":
            video_arr = _load_video_rgb_numpy(video_path, float(fps))
            t_frames = int(video_arr.shape[0])
            video_metadata = [
                {
                    "total_num_frames": t_frames,
                    "fps": float(fps),
                    "frames_indices": list(range(t_frames)),
                }
            ]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_arr},
                        {"type": "text", "text": text_prompt},
                    ],
                }
            ]
            template_kw = {
                "add_generation_prompt": True,
                "tokenize": True,
                "return_dict": True,
                "return_tensors": "pt",
                "do_sample_frames": False,
                "video_metadata": video_metadata,
            }
        else:
            raise ValueError(f"video_decode 必须是 official 或 opencv，收到: {video_decode}")

        t0 = time.time()
        inputs = self.processor.apply_chat_template(messages, **template_kw)

        target = next(self.model.parameters()).device
        if hasattr(inputs, "to"):
            inputs = inputs.to(target)
        else:
            inputs = {k: (v.to(target) if hasattr(v, "to") else v) for k, v in inputs.items()}

        pad_id = getattr(self._tokenizer, "pad_token_id", None) if self._tokenizer else None
        eos_id = getattr(self._tokenizer, "eos_token_id", None) if self._tokenizer else None

        streamer = (
            TextStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
            if self._tokenizer is not None
            else None
        )
        gen_kw: Dict[str, Any] = {
            "max_new_tokens": self._max_new_tokens,
            "do_sample": True,
            "temperature": 0.2,
            "top_p": 0.9,
            "pad_token_id": pad_id,
            "eos_token_id": eos_id,
        }
        if streamer is not None:
            gen_kw["streamer"] = streamer

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kw)

        if streamer is not None:
            print()

        in_len = inputs["input_ids"].shape[1]
        gen = out[:, in_len:]
        raw = self.processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        structured = self._parse_json(raw)
        elapsed = time.time() - t0

        return {
            "video_path": video_path,
            "raw_output": raw,
            "structured_data": structured,
            "tags": self._tags_from_structured(structured),
            "inference_time_sec": round(elapsed, 3),
            "model_path": self.model_path,
            "fps": fps,
            "video_decode": video_decode,
            "timestamp": datetime.now().isoformat(),
        }

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        text = text.strip()
        # 去掉可能的 markdown 代码块
        if "```" in text:
            m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
            if m:
                text = m.group(1).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 取第一个 JSON 对象
        start = text.find("{")
        if start == -1:
            return {}
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        return {}
        return {}

    @staticmethod
    def _tags_from_structured(d: Dict[str, Any]) -> List[str]:
        tags: List[str] = []
        seen = set()
        for key in ("objects", "scene", "person", "environment", "emotion", "color", "location"):
            v = d.get(key)
            if not v or not isinstance(v, str):
                continue
            for part in re.split(r"[,，、;；]", v):
                p = part.strip()
                if p and len(p) > 1 and p not in seen:
                    tags.append(p)
                    seen.add(p)
        return tags


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL 本地视频结构化打标")
    parser.add_argument("--video", required=True, help="视频文件路径 (.mp4 等)")
    parser.add_argument(
        "--model_path",
        default=_DEFAULT_MODEL_DIR,
        help=f"本地模型目录（默认: {_DEFAULT_MODEL_DIR}）",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="输出 JSON 路径（默认：<视频同目录>/<视频文件名>_<模型目录名>.json）",
    )
    parser.add_argument("--fps", type=float, default=1.0, help="抽帧 FPS，越低越快、越省显存（默认 1）")
    parser.add_argument(
        "--video_decode",
        choices=("official", "opencv"),
        default="official",
        help="official: 路径交给 transformers+PyAV/torchvision 解码（需 pip install av）；opencv: 本地 OpenCV 解码后喂视频数组",
    )
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="生成最大 token 数")
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None, help="强制设备")
    args = parser.parse_args()

    tagger = Qwen25VideoTagger(
        model_path=args.model_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    result = tagger.tag_video(args.video, fps=args.fps, video_decode=args.video_decode)

    if args.output is None:
        out_path = _default_output_json_path(args.video, tagger.model_path)
    else:
        out_path = os.path.abspath(os.path.normpath(args.output))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n✅ 完成，结果已写入:", out_path)
    print("structured_data:", json.dumps(result.get("structured_data"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
