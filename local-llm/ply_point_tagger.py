"""
本地点云 (.ply) 描述打标：单次生成正文描述，写入 JSON 的 description 字段；支持 PointLLM-7B-v1.2 或 GPT4Point（OPT2.7B）。

PointLLM（推荐与 video_direct_tagger 同级的「单脚本推理」体验）
  - 权重：仓库根目录 pointllm_7b_v1_2（download_model.py --model pointllm）
  - 代码：PyPI 无官方 pointllm 包；须从 GitHub 安装到当前环境（任选其一）：
      pip install "git+https://github.com/InternRobotics/PointLLM.git"
      或 git clone 后在该目录执行 pip install -e .
      若依赖想自行安装、只想注册可编辑包而不让 pip 再装一遍依赖：
      pip install -e . --no-deps
      python -m pip install "transformers @ git+https://github.com/huggingface/transformers.git@cae78c46" || python -m pip install "https://github.com/huggingface/transformers/archive/cae78c46.zip"
      pip install "torch>=2.0"
      pip install deepspeed
      pip install "tokenizers>=0.15"
      pip install open3d
      pip install scikit-learn
      pip install torchvision
      python -m pip install accelerate einops fastapi gradio "markdown2[all]" numpy requests sentencepiece
      pip install uvicorn wandb shortuuid peft openai tqdm easydict "timm==0.4.12" "ftfy==6.0.1" regex h5py termcolor plyfile nltk rouge py-rouge


GPT4Point（本脚本通过「源码仓库里的 lavis + 本地权重」推理，不是单独 pip 一个 gpt4point 包）
  1) 在本机任意目录克隆官方仓库，例如：
       git clone https://github.com/Pointcept/GPT4Point.git
     克隆后根目录下应能看到子文件夹 lavis/（Pointcept 自带魔改 LAVIS，不要装 PyPI 的 salesforce-lavis 顶替它）。
  2) 进入该目录，激活你的虚拟环境后安装依赖（与官方一致）：
       cd GPT4Point
       pip install -r requirements.txt
     若国内拉 Hugging Face 失败，可先设镜像再跑脚本：如 HF_ENDPOINT=https://hf-mirror.com
  3) 打标时把「第 1 步克隆出来的根目录」交给本脚本（二选一）：
       --gpt4point_repo D:/路径/GPT4Point
     或 PowerShell 里：$env:GPT4POINT_REPO="D:/路径/GPT4Point"
     「根目录」指包含 lavis 文件夹的那一层，不是 lavis 里面。
  4) 语言骨干 facebook/opt-2.7b（约数 GB）：无网时请放到 <weights_dir>/opt-2.7b 或设 GPT4POINT_OPT_PATH / --opt_local；
     download_model.py --model gpt4point 会尝试一并下载 bert 与 OPT。点云微调权重仍为 .pth，与 OPT 底座不同。

通用依赖：numpy torch；读点云建议 pip install open3d
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def _safe_filename_segment(name: str) -> str:
    s = re.sub(r'[<>:"/\\|?*]', "_", name.strip())
    s = s.strip(" .")
    return s or "untitled"


def _default_output_json_path(ply_path: str, model_path: str) -> str:
    ply_abs = os.path.abspath(os.path.normpath(ply_path))
    model_abs = os.path.abspath(os.path.normpath(model_path))
    ply_dir = os.path.dirname(ply_abs)
    ply_stem = os.path.splitext(os.path.basename(ply_abs))[0]
    model_name = os.path.basename(model_abs.rstrip(os.sep + (os.altsep or "")))
    base = f"{_safe_filename_segment(ply_stem)}_{_safe_filename_segment(model_name)}.json"
    return os.path.join(ply_dir, base)


_DEFAULT_POINTLLM_DIR = os.path.join(_repo_root(), "pointllm_7b_v1_2")
_DEFAULT_GPT4POINT_WEIGHTS = os.path.join(_repo_root(), "gpt4point_weight")

POINT_NUM = 8192


def _is_bert_base_snapshot(path: str) -> bool:
    """本地目录是否为 bert-base-uncased 完整快照（config + tokenizer + 权重）。"""
    if not os.path.isdir(path):
        return False
    tok = os.path.isfile(os.path.join(path, "tokenizer_config.json")) or os.path.isfile(
        os.path.join(path, "tokenizer.json")
    )
    cfg = os.path.isfile(os.path.join(path, "config.json"))
    weights = any(
        os.path.isfile(os.path.join(path, name)) for name in ("pytorch_model.bin", "model.safetensors")
    )
    return tok and cfg and weights


def _apply_gpt4point_bert_local(weights_dir: str, bert_local: Optional[str]) -> None:
    """在 import lavis 之前设置 GPT4POINT_BERT_PATH，避免离线环境访问 huggingface.co。"""
    if bert_local:
        p = os.path.abspath(os.path.normpath(bert_local))
        if not _is_bert_base_snapshot(p):
            raise FileNotFoundError(
                f"--bert_local 不是完整的 bert-base-uncased 目录（需 config.json、tokenizer_config.json、"
                f"pytorch_model.bin 或 model.safetensors）: {p}"
            )
        os.environ["GPT4POINT_BERT_PATH"] = p
        return
    if (os.environ.get("GPT4POINT_BERT_PATH") or "").strip() or (
        os.environ.get("BERT_BASE_UNCASED_PATH") or ""
    ).strip():
        return
    cand = os.path.join(weights_dir, "bert-base-uncased")
    if _is_bert_base_snapshot(cand):
        os.environ["GPT4POINT_BERT_PATH"] = os.path.abspath(cand)


def _is_opt27_snapshot(path: str) -> bool:
    """本地目录是否为 facebook/opt-2.7b 类完整快照（config + 分词器 + 权重，含分片）。"""
    if not os.path.isdir(path):
        return False
    if not os.path.isfile(os.path.join(path, "config.json")):
        return False
    root = path
    tok = os.path.isfile(os.path.join(root, "tokenizer.json")) or (
        os.path.isfile(os.path.join(root, "vocab.json"))
        and os.path.isfile(os.path.join(root, "merges.txt"))
    )
    if not tok:
        return False
    for n in os.listdir(root):
        if n.endswith(".safetensors") or n.endswith(".bin"):
            return True
    return False


def _list_facebook_opt27_hf_cache_snapshots() -> List[str]:
    """本机 HF 默认缓存中已存在的 facebook/opt-2.7b 快照目录（离线时可自动用上）。"""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        return []
    snap_root = os.path.join(HF_HUB_CACHE, "models--facebook--opt-2.7b", "snapshots")
    if not os.path.isdir(snap_root):
        return []
    out: List[str] = []
    for entry in sorted(os.listdir(snap_root)):
        p = os.path.join(snap_root, entry)
        if os.path.isdir(p):
            out.append(p)
    return out


def _omega_dotlist_path(path: str) -> str:
    """OmegaConf.from_dotlist 在 Windows 下易被反斜杠干扰，统一为正斜杠绝对路径。"""
    return Path(path).resolve().as_posix()


def _resolve_gpt4point_opt_path(weights_dir: str, opt_local: Optional[str]) -> Optional[str]:
    """返回应写入 model.opt_model 的本地目录，或 None 表示沿用 yaml 中的 Hub id。"""
    if opt_local:
        p = os.path.abspath(os.path.normpath(opt_local))
        if not _is_opt27_snapshot(p):
            raise FileNotFoundError(
                f"--opt_local 不是完整的 OPT 快照目录（需 config.json、tokenizer.json 或 vocab.json+merges.txt、"
                f"以及 *.safetensors / *.bin）: {p}"
            )
        return p
    if (os.environ.get("GPT4POINT_OPT_PATH") or "").strip() or (
        os.environ.get("FACEBOOK_OPT_2_7B_PATH") or ""
    ).strip():
        return None
    cand = os.path.join(weights_dir, "opt-2.7b")
    if _is_opt27_snapshot(cand):
        return os.path.abspath(cand)
    for snap in _list_facebook_opt27_hf_cache_snapshots():
        if _is_opt27_snapshot(snap):
            return os.path.abspath(snap)
    return None


def _make_rng(seed: Optional[int]) -> np.random.Generator:
    """seed 为 None 或负数时使用不可复现随机（与旧版行为接近）。"""
    if seed is None or seed < 0:
        return np.random.default_rng()
    return np.random.default_rng(int(seed))


# 单次纯文本描述（不要 JSON / 列表）；模型只产正文，由脚本填入下方字段壳中的 description
PLY_DESCRIPTION_PROMPT_EN = (
    "You see a colored 3D point cloud of a single object. "
    "Describe it in clear prose: what it likely is, overall shape, colors, and any salient details. "
    "Reply with plain text only (a short paragraph). Do not output JSON, markdown fences, or bullet lists."
)

PLY_DESCRIPTION_PROMPT_ZH = (
    "请根据该彩色三维点云，用一小段中文连贯描述：可能是什么物体、整体形状、颜色和显著细节。"
    "只输出普通正文段落，不要 JSON、不要 markdown、不要列表。"
)


def _empty_description_record() -> Dict[str, str]:
    return {
        "description": "",
        "object_type": "",
        "geometry_shape": "",
        "parts_structure": "",
        "color_material": "",
        "size_pose": "",
        "scene_context": "",
    }


def _finalize_description_text(text: str, max_chars: int = 8000) -> str:
    """去围栏与首尾噪声；保留多句正文（整块描述）。"""
    print(f"raw_answer={text}")
    text = (text or "").strip()
    if "```" in text:
        m = re.search(r"```(?:\w*)?\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    return text


def _default_device_dtype() -> Tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        dev = "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dev = "cpu"
        dtype = torch.float32
    return dev, dtype


def _pc_norm_pointllm(pc: np.ndarray) -> np.ndarray:
    """PointLLM 训练管线：仅对 xyz 做中心化与单位球，颜色保持 [0,1]。"""
    xyz = pc[:, :3].astype(np.float64)
    other = pc[:, 3:].astype(np.float64) if pc.shape[1] > 3 else None
    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    m = np.max(np.sqrt(np.sum(xyz**2, axis=1))) or 1.0
    xyz = xyz / m
    if other is not None:
        return np.concatenate((xyz, other), axis=1).astype(np.float32)
    return xyz.astype(np.float32)


def _farthest_point_sample(point: np.ndarray, npoint: int, rng: np.random.Generator) -> np.ndarray:
    """point: [N, D]，返回 [npoint, D]（与 PointLLM data/utils 一致）。"""
    N, D = point.shape
    if N <= npoint:
        return _upsample_points(point, npoint, rng)
    xyz = point[:, :3]
    centroids = np.zeros((npoint,), dtype=np.int64)
    distance = np.ones((N,)) * 1e10
    farthest = int(rng.integers(0, N))
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = int(np.argmax(distance))
    return point[centroids.astype(np.int32)]


def _upsample_points(point: np.ndarray, npoint: int, rng: np.random.Generator) -> np.ndarray:
    if point.shape[0] == npoint:
        return point
    if point.shape[0] == 0:
        raise ValueError("点云为空")
    idx = rng.choice(point.shape[0], npoint, replace=True)
    return point[idx]


def _pc_norm_gpt4point(pc: np.ndarray) -> np.ndarray:
    """GPT4Point eval：与 lavis.datasets.transforms.transforms_point.pc_norm_with_color 一致。"""
    if pc.shape[1] == 3:
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1))) or 1.0
        return (pc / m).astype(np.float32)
    assert pc.shape[1] == 6
    pc_xyz = pc[:, :3].astype(np.float64)
    centroid = np.mean(pc_xyz, axis=0)
    pc_xyz = pc_xyz - centroid
    m = np.max(np.sqrt(np.sum(pc_xyz**2, axis=1))) or 1.0
    pc_xyz = pc_xyz / m
    pc_color = pc[:, 3:6].astype(np.float64)
    pc_color = (pc_color - 0.5) * 2.0
    out = np.concatenate((pc_xyz, pc_color), axis=1).astype(np.float32)
    return out


def _load_ply_xyzrgb(path: str) -> np.ndarray:
    """
    返回 float32 数组 [N, 6]，RGB 在 [0,1]；若无颜色则 RGB 填 0.5。
    """
    path = os.path.abspath(os.path.normpath(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"点云不存在: {path}")

    try:
        import open3d as o3d  # type: ignore

        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points, dtype=np.float64)
        if pts.size == 0:
            raise RuntimeError(f"未读取到顶点: {path}")
        colors = np.asarray(pcd.colors, dtype=np.float64) if pcd.has_colors() else None
    except ImportError:
        pts, colors = _load_ply_ascii_fallback(path)

    n = pts.shape[0]
    if colors is None or colors.shape[0] != n:
        colors = np.full((n, 3), 0.5, dtype=np.float64)
    else:
        if colors.max() > 1.0 + 1e-3:
            colors = np.clip(colors / 255.0, 0.0, 1.0)
        else:
            colors = np.clip(colors, 0.0, 1.0)

    return np.concatenate([pts.astype(np.float32), colors.astype(np.float32)], axis=1)


def _load_ply_ascii_fallback(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """最小 ASCII PLY 读取（无 open3d 时的退路）。"""
    vertices: List[List[float]] = []
    colors: List[List[float]] = []
    format_ascii = False
    n_vert = 0
    props: List[str] = []
    in_header = True
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if in_header:
                if line.startswith("format ascii"):
                    format_ascii = True
                elif line.startswith("element vertex"):
                    n_vert = int(line.split()[-1])
                elif line.startswith("property"):
                    props.append(line.split()[-1])
                elif line == "end_header":
                    in_header = False
                    if not format_ascii:
                        raise RuntimeError("无 open3d 时仅支持 ASCII PLY，请 pip install open3d")
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
            if len(parts) >= 6 and all(p in props for p in ("red", "green", "blue")):
                colors.append([float(parts[3]) / 255.0, float(parts[4]) / 255.0, float(parts[5]) / 255.0])
            if len(vertices) >= n_vert:
                break
    pts_arr = np.array(vertices, dtype=np.float64)
    col_arr = np.array(colors, dtype=np.float64) if colors else None
    return pts_arr, col_arr


def _tags_from_structured(d: Dict[str, Any]) -> List[str]:
    tags: List[str] = []
    seen = set()
    for key in (
        "object_type",
        "geometry_shape",
        "parts_structure",
        "color_material",
        "scene_context",
    ):
        v = d.get(key)
        if not v or not isinstance(v, str):
            continue
        for part in re.split(r"[,，、;；]", v):
            p = part.strip()
            if p and len(p) > 1 and p not in seen:
                tags.append(p)
                seen.add(p)
    return tags


def _import_gpt4point_lavis() -> None:
    """加载 Pointcept/GPT4Point 内嵌 lavis 子包，完成 registry 注册（与 evaluate.py 顺序一致）。"""
    import lavis.tasks  # noqa: F401
    import lavis.datasets.builders  # noqa: F401
    import lavis.models  # noqa: F401
    import lavis.processors  # noqa: F401


class PointLLMPointTagger:
    """依赖可导入的 pointllm 包（见文件头：pip install git+… 或 clone 后 pip install -e .）。"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        max_new_tokens: int = 1024,
        torch_dtype: Optional[str] = None,
    ):
        try:
            from pointllm.conversation import SeparatorStyle, conv_templates
            from pointllm.model import PointLLMLlamaForCausalLM
            from pointllm.model.utils import KeywordsStoppingCriteria
            from pointllm.utils import disable_torch_init
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "未找到 pointllm 包。请任选其一安装（需联网、版本以仓库为准）：\n"
                '  pip install "git+https://github.com/InternRobotics/PointLLM.git"\n'
                "或 git clone 该仓库后在目录内执行: pip install -e ."
            ) from e

        self._conv_templates = conv_templates
        self._SeparatorStyle = SeparatorStyle
        self._KeywordsStoppingCriteria = KeywordsStoppingCriteria
        self._disable_torch_init = disable_torch_init

        self.model_path = os.path.abspath(os.path.normpath(model_path or _DEFAULT_POINTLLM_DIR))
        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(
                f"未找到本地模型目录: {self.model_path}\n请先运行: python local-llm/download_model.py --model pointllm"
            )

        if device:
            self.device = device
            if torch_dtype == "float32" or device == "cpu":
                self.dtype = torch.float32
            elif torch_dtype == "float16":
                self.dtype = torch.float16
            elif torch_dtype == "bfloat16":
                self.dtype = torch.bfloat16
            else:
                _, self.dtype = _default_device_dtype()
        else:
            self.device, self.dtype = _default_device_dtype()

        self._disable_torch_init()
        print(f"加载 PointLLM: {self.model_path}")
        print(f"设备: {self.device}, dtype: {self.dtype}")

        tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        load_kw: Dict[str, Any] = {
            "low_cpu_mem_usage": True,
            "use_cache": True,
            "torch_dtype": self.dtype,
        }
        if self.device == "cpu":
            load_kw["device_map"] = None
        else:
            load_kw["device_map"] = "auto"

        self.model = PointLLMLlamaForCausalLM.from_pretrained(self.model_path, **load_kw)
        self.model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
        self.model.eval()
        self._tokenizer = tokenizer
        self._max_new_tokens = max_new_tokens

        mm_use_point_start_end = getattr(self.model.config, "mm_use_point_start_end", False)
        point_backbone_config = self.model.get_model().point_backbone_config
        conv = self._conv_templates["vicuna_v1_1"].copy()
        stop_str = conv.sep if conv.sep_style != self._SeparatorStyle.TWO else conv.sep2
        self._mm_use_point_start_end = mm_use_point_start_end
        self._point_backbone_config = point_backbone_config
        self._conv_template = conv
        self._stop_keywords = [stop_str]

    def _infer_one_question(
        self,
        point_clouds: torch.Tensor,
        user_question: str,
        device_t: torch.device,
        *,
        temperature: float,
        seed: Optional[int],
        max_new_tokens: int,
    ) -> str:
        conv = self._conv_template.copy()
        conv.reset()
        point_token_len = self._point_backbone_config["point_token_len"]
        default_point_patch_token = self._point_backbone_config["default_point_patch_token"]
        default_point_start_token = self._point_backbone_config["default_point_start_token"]
        default_point_end_token = self._point_backbone_config["default_point_end_token"]
        qs = user_question
        if self._mm_use_point_start_end:
            qs = (
                default_point_start_token
                + default_point_patch_token * point_token_len
                + default_point_end_token
                + "\n"
                + qs
            )
        else:
            qs = default_point_patch_token * point_token_len + "\n" + qs
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()
        inputs = self._tokenizer([prompt_text])
        input_ids = torch.as_tensor(inputs.input_ids).to(device_t)

        stopping_criteria = self._KeywordsStoppingCriteria(self._stop_keywords, self._tokenizer, input_ids)

        pad_id = getattr(self._tokenizer, "pad_token_id", None)
        eos_id = getattr(self._tokenizer, "eos_token_id", None)
        do_sample = float(temperature) > 0.0
        gen_kw: Dict[str, Any] = {
            "do_sample": do_sample,
            "max_new_tokens": max_new_tokens,
            "stopping_criteria": [stopping_criteria],
            "pad_token_id": pad_id,
            "eos_token_id": eos_id,
        }
        if do_sample:
            gen_kw["temperature"] = float(temperature)
            gen_kw["top_p"] = 0.9
            if seed is not None and seed >= 0:
                dev = device_t.type if hasattr(device_t, "type") else str(device_t)
                g = torch.Generator(device="cuda" if dev == "cuda" else "cpu")
                g.manual_seed(int(seed))
                gen_kw["generator"] = g

        with torch.inference_mode():
            try:
                output_ids = self.model.generate(
                    input_ids,
                    point_clouds=point_clouds,
                    **gen_kw,
                )
            except TypeError:
                gen_kw.pop("max_new_tokens", None)
                gen_kw["max_length"] = int(input_ids.shape[1] + max_new_tokens)
                output_ids = self.model.generate(
                    input_ids,
                    point_clouds=point_clouds,
                    **gen_kw,
                )

        in_len = input_ids.shape[1]
        raw = self._tokenizer.batch_decode(output_ids[:, in_len:], skip_special_tokens=True)[0].strip()
        stop = self._stop_keywords[0]
        if raw.endswith(stop):
            raw = raw[: -len(stop)].strip()
        return raw

    def tag_ply(
        self,
        ply_path: str,
        point_num: int = POINT_NUM,
        seed: Optional[int] = 42,
        temperature: float = 0.2,
        qa_max_new_tokens: int = 512,
        prompt_lang: str = "en",
        user_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """单次生成点云描述；结果的 structured_data 为固定七键，仅 description 有正文，其余为空串。"""
        if seed is not None and seed >= 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        rng = _make_rng(seed)
        raw_pc = _load_ply_xyzrgb(ply_path)
        pc = _farthest_point_sample(raw_pc.astype(np.float64), point_num, rng)
        pc = _pc_norm_pointllm(pc)
        point_clouds = torch.from_numpy(pc).unsqueeze(0).to(torch.float32)
        device_t = next(self.model.parameters()).device
        point_clouds = point_clouds.to(device_t)

        t0 = time.time()
        question = (user_prompt or "").strip()
        if not question:
            question = (
                PLY_DESCRIPTION_PROMPT_ZH if prompt_lang == "zh" else PLY_DESCRIPTION_PROMPT_EN
            )
        raw_answer = self._infer_one_question(
            point_clouds,
            question,
            device_t,
            temperature=temperature,
            seed=seed,
            max_new_tokens=qa_max_new_tokens,
        )
        desc = _finalize_description_text(raw_answer)
        record = _empty_description_record()
        record["description"] = desc

        elapsed = time.time() - t0
        return {
            "ply_path": os.path.abspath(ply_path),
            "raw_output": raw_answer,
            "structured_data": record,
            "tags": _tags_from_structured(record),
            "inference_time_sec": round(elapsed, 3),
            "model_path": self.model_path,
            "model_type": "pointllm",
            "num_points_used": point_num,
            "timestamp": datetime.now().isoformat(),
        }


class GPT4PointPLYTagger:
    """
    使用 Pointcept/GPT4Point 仓内 lavis 加载 GPT4Point_OPT；权重默认来自本仓库 gpt4point 目录。
    """

    def __init__(
        self,
        weights_dir: Optional[str] = None,
        gpt4point_repo: Optional[str] = None,
        device: Optional[str] = None,
        max_gen_length: int = 256,
        min_gen_length: int = 8,
        num_beams: int = 5,
        bert_local: Optional[str] = None,
        opt_local: Optional[str] = None,
    ):
        weights_dir = os.path.abspath(os.path.normpath(weights_dir or _DEFAULT_GPT4POINT_WEIGHTS))
        _apply_gpt4point_bert_local(weights_dir, bert_local)
        opt_path = _resolve_gpt4point_opt_path(weights_dir, opt_local)
        repo = gpt4point_repo or os.environ.get("GPT4POINT_REPO")
        if not repo or not os.path.isdir(repo):
            raise FileNotFoundError(
                "GPT4Point 需本地克隆 Pointcept/GPT4Point 源码，并通过 --gpt4point_repo 或环境变量 GPT4POINT_REPO "
                "指定仓库根目录（内含 lavis/）。示例：git clone https://github.com/Pointcept/GPT4Point.git"
            )
        repo = os.path.abspath(repo)
        finetuned = os.path.join(weights_dir, "gpt4point_pretrain_stage2_opt2.7b.pth")
        point_enc = os.path.join(weights_dir, "point_encoder_pointbert_wcolor.pth")
        for p, label in ((finetuned, "stage2 权重"), (point_enc, "点云编码器")):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"缺少 {label}: {p}\n请运行: python local-llm/download_model.py --model gpt4point")

        if repo not in sys.path:
            sys.path.insert(0, repo)

        _import_gpt4point_lavis()
        from lavis.common.config import Config
        from lavis.common.registry import registry

        cfg_yml = os.path.join(repo, "lavis", "projects", "gpt4point", "eval", "captioning3d_cap3d_opt2.7b_eval.yaml")
        if not os.path.isfile(cfg_yml):
            raise FileNotFoundError(f"未找到评测配置: {cfg_yml}")

        cfg_options = [
            f"model.finetuned={_omega_dotlist_path(finetuned)}",
            f"model.point_encoder_cfg.checkpoint={_omega_dotlist_path(point_enc)}",
            # eval yaml 曾写空列表，会导致 load_checkpoint 要求 ckpt 含 point_encoder/opt 全部参数而报错
            "model.ckpt_special_strs=[opt_model,point_encoder,opt_proj]",
            "run.distributed=False",
            f"run.device={'cuda' if (device or ('cuda' if torch.cuda.is_available() else 'cpu')) == 'cuda' else 'cpu'}",
        ]
        if opt_path:
            cfg_options.append(f"model.opt_model={_omega_dotlist_path(opt_path)}")
            print(f"GPT4Point: 使用本地 OPT 目录: {_omega_dotlist_path(opt_path)}")
        elif not (os.environ.get("GPT4POINT_OPT_PATH") or os.environ.get("FACEBOOK_OPT_2_7B_PATH") or "").strip():
            print(
                "GPT4Point: 未找到本地 OPT（请准备 gpt4point_weight/opt-2.7b 或 HF 缓存 snapshot）。"
                "将尝试联网加载 facebook/opt-2.7b；若失败请运行: python local-llm/download_model.py --model gpt4point"
            )

        ns = SimpleNamespace(
            cfg_path=cfg_yml,
            options=cfg_options,
            local_rank=0,
        )
        cfg = Config(ns)
        model = registry.get_model_class(cfg.model_cfg.arch).from_config(cfg.model_cfg)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
        self.device = dev
        self.model = model.to(torch.device(dev))
        self.model.eval()
        self._weights_dir = weights_dir
        self._repo = repo
        self._max_gen_length = max_gen_length
        self._min_gen_length = min_gen_length
        self._num_beams = num_beams
        print(f"加载 GPT4Point（lavis），权重目录: {weights_dir}，代码: {repo}")

    def tag_ply(
        self,
        ply_path: str,
        prompt: Optional[str] = None,
        point_num: int = POINT_NUM,
        seed: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """单次 caption；structured_data 为固定七键，仅 description 为模型正文，其余为空串。"""
        rng = _make_rng(seed)
        raw_pc = _load_ply_xyzrgb(ply_path)
        pc = _farthest_point_sample(raw_pc.astype(np.float64), point_num, rng)
        pc = _pc_norm_gpt4point(pc)
        point_batch = torch.from_numpy(pc).unsqueeze(0).to(torch.float32).to(self.model.device)

        text_instruction = (prompt or "").strip() or PLY_DESCRIPTION_PROMPT_EN
        samples: Dict[str, Any] = {
            "point": point_batch,
            "pcd_id": [os.path.basename(ply_path)],
            "text_input": [text_instruction],
        }
        t0 = time.time()
        with torch.no_grad():
            caps = self.model.generate(
                samples,
                use_nucleus_sampling=False,
                num_beams=self._num_beams,
                max_length=self._max_gen_length,
                min_length=self._min_gen_length,
            )
        raw = caps[0] if caps else ""
        desc = _finalize_description_text(raw)
        structured = _empty_description_record()
        structured["description"] = desc
        elapsed = time.time() - t0
        return {
            "ply_path": os.path.abspath(ply_path),
            "raw_output": raw,
            "structured_data": structured,
            "tags": _tags_from_structured(structured),
            "inference_time_sec": round(elapsed, 3),
            "model_path": self._weights_dir,
            "model_type": "gpt4point",
            "gpt4point_repo": self._repo,
            "num_points_used": point_num,
            "timestamp": datetime.now().isoformat(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="本地 .ply 点云描述打标（PointLLM / GPT4Point）")
    parser.add_argument("--ply", required=True, help="PLY 文件路径")
    parser.add_argument(
        "--model_type",
        choices=("pointllm", "gpt4point"),
        default="pointllm",
        help="pointllm: PointLLM-7B-v1.2（需从 GitHub 安装 pointllm，如 pip install git+https://github.com/InternRobotics/PointLLM.git）；"
        "gpt4point: 需克隆 Pointcept/GPT4Point 并设置 --gpt4point_repo",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="PointLLM 时：本地模型目录（默认 pointllm_7b_v1_2）；GPT4Point 时：可省略，改用 --weights_dir",
    )
    parser.add_argument(
        "--weights_dir",
        default=_DEFAULT_GPT4POINT_WEIGHTS,
        help=f"GPT4Point 权重目录（默认: {_DEFAULT_GPT4POINT_WEIGHTS}）",
    )
    parser.add_argument(
        "--gpt4point_repo",
        default=None,
        help="Pointcept/GPT4Point 仓库根目录；也可设置环境变量 GPT4POINT_REPO",
    )
    parser.add_argument(
        "--bert_local",
        default=None,
        help="GPT4Point：bert-base-uncased 本地目录；不设则用环境变量 GPT4POINT_BERT_PATH / BERT_BASE_UNCASED_PATH，"
        "或自动使用 <weights_dir>/bert-base-uncased",
    )
    parser.add_argument(
        "--opt_local",
        default=None,
        help="GPT4Point：facebook/opt-2.7b 本地目录；不设则用 GPT4POINT_OPT_PATH / FACEBOOK_OPT_2_7B_PATH，"
        "或自动使用 <weights_dir>/opt-2.7b（download_model.py 会下载到此路径）",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="输出 JSON（默认：<ply 同目录>/<ply 名>_<模型标识>.json）",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="自定义单次描述用提示语（覆盖默认英文/中文描述模板）",
    )
    parser.add_argument(
        "--prompt_lang",
        choices=("en", "zh"),
        default="en",
        help="未指定 --prompt 时的描述模板语言；PointLLM 底座为英文 Vicuna，中文可能不稳定，默认 en",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="最远点采样等随机性种子；默认 42 可复现；-1 表示每次随机",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="仅 PointLLM：>0 时启用采样（含随机性）；默认 0 为贪心解码，输出更稳定",
    )
    parser.add_argument(
        "--qa_max_new_tokens",
        type=int,
        default=512,
        help="仅 PointLLM：描述生成的 max_new_tokens",
    )
    parser.add_argument("--point_num", type=int, default=POINT_NUM, help=f"采样点数（默认 {POINT_NUM}）")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="PointLLM 生成上限")
    parser.add_argument("--max_gen_length", type=int, default=256, help="GPT4Point generate 的 max_length")
    parser.add_argument("--min_gen_length", type=int, default=8, help="GPT4Point generate 的 min_length")
    parser.add_argument("--num_beams", type=int, default=5, help="GPT4Point beam size")
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument(
        "--torch_dtype",
        choices=("float32", "float16", "bfloat16"),
        default=None,
        help="仅 PointLLM：覆盖默认 dtype（GPU 默认 fp16/bf16）",
    )
    args = parser.parse_args()

    seed_val: Optional[int] = None if args.seed < 0 else args.seed
    user_prompt = (args.prompt or "").strip() or None
    default_desc = (
        PLY_DESCRIPTION_PROMPT_ZH if args.prompt_lang == "zh" else PLY_DESCRIPTION_PROMPT_EN
    )
    gpt4point_instruction = user_prompt or default_desc

    if args.model_type == "pointllm":
        mp = args.model_path or _DEFAULT_POINTLLM_DIR
        tagger = PointLLMPointTagger(
            model_path=mp,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            torch_dtype=args.torch_dtype,
        )
        result = tagger.tag_ply(
            args.ply,
            point_num=args.point_num,
            seed=seed_val,
            temperature=args.temperature,
            qa_max_new_tokens=args.qa_max_new_tokens,
            prompt_lang=args.prompt_lang,
            user_prompt=user_prompt,
        )
        model_ref = tagger.model_path
    else:
        tagger = GPT4PointPLYTagger(
            weights_dir=args.weights_dir,
            gpt4point_repo=args.gpt4point_repo,
            device=args.device,
            max_gen_length=args.max_gen_length,
            min_gen_length=args.min_gen_length,
            num_beams=args.num_beams,
            bert_local=args.bert_local,
            opt_local=args.opt_local,
        )
        result = tagger.tag_ply(
            args.ply, prompt=gpt4point_instruction, point_num=args.point_num, seed=seed_val
        )
        model_ref = tagger._weights_dir

    if args.output is None:
        out_path = _default_output_json_path(args.ply, model_ref)
    else:
        out_path = os.path.abspath(os.path.normpath(args.output))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n✅ 完成，结果已写入:", out_path)
    print("structured_data:", json.dumps(result.get("structured_data"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
