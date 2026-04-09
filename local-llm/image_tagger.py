import torch
from PIL import Image, ImageDraw, ImageFont
import os
import re
import argparse
from typing import List, Dict, Tuple, Optional, Any, Set
import json
import time
from datetime import datetime
from transformers import TextStreamer


# DeepSeek-VL 官方视觉定位：坐标 0～999，像素 = coord / 999 * 宽高
DEEPSEEK_GROUNDING_PROMPT = """请根据上图，用 DeepSeek-VL 视觉定位格式输出图中主要目标（人物、动物、车辆、建筑等显著物体）的轴对齐矩形框。
每个目标单独一行，格式必须一字不差地使用尖括号与竖线（不要写成 |ref|，必须写成 <|ref|>）：
<|ref|>中文名称<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>
x1,y1,x2,y2 均为 0～999 的整数：原点在左上角，x 向右、y 向下；(x1,y1) 为左上角，(x2,y2) 为右下角。

重要：**人物/人体**的框必须从**头顶（含帽子/头发）最上缘**到**脚底/鞋底最下缘**，左右包含**展臂**；禁止只框躯干或上半身。动物、车辆同理须包全可见部分，不要用小块区域代替整目标。
"""


# 用于合并 det 与 JSON 物体、以及相对框外扩提示
PERSON_LABEL_HINT = re.compile(
    r"人|男|女|孩|跑者|选手|运动员|肖像|主角|行人|人体|脸部|人脸|少年|青年|老人"
)


class DeepSeekImageTagger:
    def __init__(
        self,
        model_path=None,
        device=None,
        max_new_tokens=256,
        model_type="deepseek-vl",
        grounding_second_pass: bool = True,
        expand_person_bbox: bool = True,
    ):
        """初始化图像打标系统。device 可选 'cuda'/'cpu'，不传则自动检测。
        
        Args:
            model_path: 模型路径；deepseek-vl 未指定本地目录时默认拉取 HF `deepseek-vl-7b-chat`；blip2 须指定本地路径
            device: 计算设备 'cuda' 或 'cpu'
            max_new_tokens: 生成最大 token 数
            model_type: 模型类型，'deepseek-vl' 或 'blip2'
            grounding_second_pass: DeepSeek-VL 是否在 JSON 描述后再跑一轮 <|ref|><|det|> 定位（0～999）
            expand_person_bbox: 人物类目标在 0～1 相对 bbox 上适度外扩（从头到脚更易框全）
        """
        print(f"🚀 初始化图像打标系统...")
        self._max_new_tokens = max_new_tokens
        self.model_path = model_path
        self.model_type = model_type
        self._grounding_second_pass = grounding_second_pass and model_type == "deepseek-vl"
        self._expand_person_bbox = expand_person_bbox

        # 提取模型名称（用于文件名）
        self.model_name_suffix = self._extract_model_name(model_path)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        # GPU 用半精度加速，CPU 用 float32
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        if self.device == "cuda" and hasattr(torch, "bfloat16"):
            try:
                self.dtype = torch.bfloat16  # 更稳，显存友好
            except Exception:
                pass
        print(f"使用设备: {self.device} (dtype={self.dtype})")
        print(f"模型类型: {model_type}")

        if model_type == "blip2":
            # 直接使用 BLIP-2 模型
            self._load_blip2_model(model_path)
        else:
            # 默认使用 DeepSeek-VL
            self._load_deepseek_vl_model(model_path)

    def _load_deepseek_vl_model(self, model_path):
        """加载 DeepSeek-VL 模型"""
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            # Chat 版更适合对话式提示与官方 <|ref|><|det|> 定位；本地目录仍优先
            default_hf = "deepseek-community/deepseek-vl-7b-chat"
            if model_path:
                model_path_abs = os.path.abspath(os.path.normpath(model_path))
                if os.path.isdir(model_path_abs):
                    model_name = model_path_abs
                else:
                    model_name = default_hf
            else:
                model_name = default_hf

            print(f"加载 DeepSeek-VL 模型: {model_name}")

            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            device_map = "cuda" if self.device == "cuda" else "cpu"
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                trust_remote_code=True,
                dtype=self.dtype,
                low_cpu_mem_usage=True,
                device_map=device_map,
            )
            self.model.eval()
            self.model_type = "deepseek-vl"
            
        except Exception as e:
            print(f"DeepSeek-VL加载失败: {e}")
            raise  # 直接抛出异常，不再自动回退

    def _load_blip2_model(self, model_path):
        """加载指定路径的 BLIP-2 模型"""
        from transformers import Blip2Processor, Blip2ForConditionalGeneration

        if model_path and os.path.isdir(model_path):
            model_name = os.path.abspath(os.path.normpath(model_path))
            print(f"加载本地 BLIP-2 模型: {model_name}")
        else:
            raise ValueError("使用 BLIP-2 模型时必须指定有效的本地模型路径 (--model_path)")

        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        self.model.eval()
        self.model_type = "blip2"

    def _deepseek_multimodal_generate(
        self,
        image: Image.Image,
        prompt_text: str,
        max_new_tokens: Optional[int] = None,
        use_streamer: bool = True,
    ) -> str:
        """图 + 文 -> 解码文本（不含 prompt 前缀剔除逻辑在调用处用 prompt_text 做 strip）。"""
        import base64
        from io import BytesIO

        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{img_base64}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image", "url": data_url},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = getattr(self.model, "device", torch.device("cpu"))
        dtype = getattr(self.model, "dtype", torch.float32)
        if hasattr(dtype, "dtype"):
            dtype = dtype.dtype
        if hasattr(inputs, "to"):
            inputs = inputs.to(device, dtype=dtype)
        else:
            inputs = {k: (v.to(device, dtype=dtype) if hasattr(v, "to") else v) for k, v in inputs.items()}

        tokenizer = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else None
        pad_token_id = tokenizer.pad_token_id if tokenizer and hasattr(tokenizer, "pad_token_id") else None
        eos_token_id = tokenizer.eos_token_id if tokenizer and hasattr(tokenizer, "eos_token_id") else None

        streamer = None
        if use_streamer and tokenizer:
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        cap = max_new_tokens if max_new_tokens is not None else min(getattr(self, "_max_new_tokens", 1024), 1024)
        # 无 streamer（如第二路 <|det|>）用贪心，坐标更稳
        do_sample = bool(streamer)
        gen_kw = dict(
            max_new_tokens=cap,
            streamer=streamer,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        if do_sample:
            gen_kw["do_sample"] = True
            gen_kw["temperature"] = 0.2
            gen_kw["top_p"] = 0.9
        else:
            gen_kw["do_sample"] = False
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kw)
        input_length = inputs["input_ids"].shape[1]
        generated_ids_trimmed = [out_ids[input_length:] for out_ids in generated_ids]
        decoded = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        raw_output = decoded[0] if decoded else ""

        _prompt = (prompt_text or "").strip()
        if _prompt and len(_prompt) > 20:
            _prefix = _prompt[: min(80, len(_prompt))]
            if raw_output.strip().startswith(_prefix):
                if raw_output.startswith(_prompt):
                    raw_output = raw_output[len(_prompt) :].strip()
                else:
                    for i in range(min(len(_prompt), len(raw_output)), 0, -1):
                        if raw_output.startswith(_prompt[:i]):
                            raw_output = raw_output[i:].strip()
                            break
        return raw_output

    def _det_bbox_plausible_999(self, bbox: List[int]) -> bool:
        """过滤明显无效的 det（如 36×36 点块），避免覆盖 JSON 好框。"""
        if len(bbox) < 4:
            return False
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        w, h = abs(x2 - x1), abs(y2 - y1)
        if w < 48 or h < 48:
            return False
        if w * h < 55 * 55:
            return False
        return True

    def _parse_ref_det_output(self, text: str) -> List[Dict[str, Any]]:
        """解析官方 <|ref|>…<|det|>… 及常见变体（如 |ref|…|det|…），0～999 grid。"""
        out: List[Dict[str, Any]] = []
        if not text:
            return out
        seen_spans: Set[Tuple[int, int]] = set()

        def try_append(label: str, det_raw: str, start: int, end: int) -> None:
            key = (start, end)
            if key in seen_spans:
                return
            label = (label or "").strip()
            det_raw = (det_raw or "").strip()
            try:
                arr = json.loads(det_raw)
            except json.JSONDecodeError:
                return
            if isinstance(arr, list) and arr:
                inner = arr[0] if isinstance(arr[0], list) else arr
                if isinstance(inner, (list, tuple)) and len(inner) >= 4:
                    bbox = [int(round(float(inner[i]))) for i in range(4)]
                    bbox = [max(0, min(999, v)) for v in bbox]
                    if not self._det_bbox_plausible_999(bbox):
                        return
                    seen_spans.add(key)
                    out.append({"label": label, "bbox": bbox, "coord_space": "999"})

        patterns: List[Tuple[str, int]] = [
            # 官方
            (
                r"<\|ref\|>(.*?)<\|/ref\|>\s*<\|det\|>\s*(.+?)\s*<\|/det\|>",
                0,
            ),
            # 模型常漏尖括号：|ref|名称|det|[[..]]
            (
                r"\|ref\|([^|]+?)\|det\|(\[\[.+?\]\])",
                0,
            ),
            # 带 < 仅半边等：宽松
            (
                r"<\|ref\|>(.*?)<\|/ref\|>\s*<\|det\|>\s*(\[\[.+?\]\])",
                0,
            ),
        ]
        for pat, _ in patterns:
            for m in re.finditer(pat, text, re.DOTALL | re.IGNORECASE):
                try_append(m.group(1), m.group(2), m.start(), m.end())
        return out

    def _labels_person_compatible(self, a: str, b: str) -> bool:
        a, b = (a or "").strip(), (b or "").strip()
        if not a or not b:
            return False
        if PERSON_LABEL_HINT.search(a) and PERSON_LABEL_HINT.search(b):
            return True
        return False

    def _merge_ref_det_into_objects(self, structured_data: Dict, detections: List[Dict[str, Any]]) -> None:
        if not detections:
            return
        objs = structured_data.get("objects")
        if not isinstance(objs, list):
            objs = []
            structured_data["objects"] = objs
        for det in detections:
            label = (det.get("label") or "").strip()
            bbox = det.get("bbox")
            if not label or not isinstance(bbox, list) or len(bbox) < 4:
                continue
            merged = False
            for obj in objs:
                if not isinstance(obj, dict):
                    continue
                ol = (obj.get("label") or obj.get("name") or "").strip()
                if (
                    ol == label
                    or (ol and (ol in label or label in ol))
                    or self._labels_person_compatible(ol, label)
                ):
                    obj["bbox"] = bbox
                    obj["coord_space"] = "999"
                    merged = True
                    break
            if not merged:
                objs.append({"label": label, "bbox": bbox, "coord_space": "999"})

    def _expand_person_bbox_rel_on_objects(self, structured_data: Dict) -> None:
        """对 0～1 相对 bbox 且含人物语义的条目适度外扩（不修改 grid_999）。"""
        if not getattr(self, "_expand_person_bbox", True):
            return
        objs = structured_data.get("objects")
        if not isinstance(objs, list):
            return
        for obj in objs:
            if not isinstance(obj, dict):
                continue
            label = (obj.get("label") or obj.get("name") or "").strip()
            if not PERSON_LABEL_HINT.search(label):
                continue
            if (obj.get("coord_space") or "") == "999":
                continue
            bbox = obj.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            try:
                x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            except (TypeError, ValueError):
                continue
            if max(x0, y0, x1, y1) > 1.02:
                continue
            h = y1 - y0
            if h >= 0.78:
                continue
            x0 = max(0.0, x0 - 0.10)
            x1 = min(1.0, x1 + 0.10)
            y0 = max(0.0, y0 - 0.14)
            y1 = min(1.0, y1 + 0.12)
            obj["bbox"] = [x0, y0, x1, y1]
            obj["bbox_expanded_heuristic"] = True

    def analyze_image(self, image_path: str, prompt: str = None) -> Dict:
        """分析图像并生成描述和标签"""
        print(f"分析图像: {image_path}")

        # 打开图像
        image = Image.open(image_path).convert("RGB")

        start_time = time.time()

        # 定义结构化输出的JSON格式提示（objects：优先 bbox 轴对齐框，坐标 0-1000）
        structured_prompt_template = """你是一个图片分析助手，你的任务是根据下面图片内容，生成一个结构化描述，只输出一个 JSON 对象（不要 Markdown、不要代码块）：
{"description": "", "scene": "", "objects": [], "color": "", "person": "", "environment": "", "emotion": "", "location": ""}

字段说明：
- description: 对图片的详细文字描述，含画面内容、构图、整体氛围（50-200字）
- scene: 用一句话概括场景
- objects: 主要物体（含人物、显著物品）的数组。每项优先提供轴对齐包围框，格式为：
  {"label": "名称", "bbox": [x_min, y_min, x_max, y_max]}
  - 优先使用 0～1000 的整数网格（与整张图对齐）。也可用小数 0～1 表示相对宽高（程序会自动识别）。
  - bbox 表示紧贴该物体**完整可见区域**的最小水平竖直矩形：须保证 **x_min < x_max、y_min < y_max**。
  - **人物**：框须为**全身**：y 方向从**头顶/帽子最上**到**脚底/鞋底最下**，x 方向含**展臂**最宽处；勿只框躯干。若用 0～1 相对坐标，完整站立/跑步人体在画面中常见 **(y_max - y_min) 约 0.55～0.92**，过小则框偏紧。
  - **非人物**：框须包住整个物体，不要只标物体旁的空地或阴影。
  - 若一时给不出 bbox，可改用 "points"：单点 [[x,y]] 表示中心；或四点按顺序 左上、右上、右下、左下（须包住整块物体，不要四点挤在一小块地面）。
  - 完全无法定位时可省略该项或 "bbox"/"points" 为空。
- color: 图片中的主要颜色
- person: 人物描述（动作、所在位置、穿着），无人或看不清时填 null，看不清穿着可写「看不清」
- environment: 户外环境（天气、天空、时间、体感温度等），室内或无法判断时填 null
- emotion: 图片传达的情绪或氛围
- location: 场所类型（户外/室内/庭院/办公室等）
"""

        if self.model_type == "deepseek-vl":
            if prompt is None:
                prompt = structured_prompt_template
            print("✅ 使用 base64 data URL 加载图片到 DeepSeek-VL")
            max_tokens = min(getattr(self, "_max_new_tokens", 1024), 1024)
            raw_output = self._deepseek_multimodal_generate(
                image, prompt, max_new_tokens=max_tokens, use_streamer=True
            )
            print(
                f"📝 DeepSeek-VL 生成的原始描述: {raw_output[:500]}..."
                if len(raw_output) > 500
                else f"📝 DeepSeek-VL 生成的原始描述: {raw_output}"
            )
            result = raw_output

        else:  # BLIP-2
            # BLIP-2 支持 caption（不传 text）和 VQA（传 text 作为问题）
            inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float32)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=100)

            caption = self.processor.decode(outputs[0], skip_special_tokens=True).strip()
            # VQA 必须使用官方格式 "Question: {问题} Answer:"，模型才会只生成答案
            # 见 https://huggingface.co/blog/blip-2
            # vqa_question = prompt if isinstance(prompt, str) and prompt.strip() else None
            # VQA 问题与输出字段一一对应，便于直接构建结构化 JSON（value 为英文）
            vqa_questions = [
                ("scene", "What is the main scene or activity?"),
                ("description", "Give a brief overall description of this image in one or two sentences."),
                ("environment", "Is it indoors or outdoors? What is the weather and environment?"),
                ("objects", "What are the main objects in this image?"),
                ("person", "What is the person doing? Describe the person if any."),
                ("location", "Where is this? What is the location?"),
                ("emotion", "What is the mood or emotion portrayed?"),
                ("color", "What are the main colors in this image?"),
            ]

            answers = {}
            for field, q in vqa_questions:
                vqa_prompt = f"Question: {q.strip()} Answer:"
                inputs = self.processor(image, text=vqa_prompt, return_tensors="pt").to(self.device, torch.float32)
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=100)
                ans = self.processor.decode(outputs[0], skip_special_tokens=True).strip()
                # 只保留 Answer: 后面的内容
                if "Answer:" in ans:
                    ans = ans.split("Answer:", 1)[1].strip()
                answers[field] = ans

            description = answers.get("description", "")
            structured = {
                "scene": answers.get("scene", ""),
                "description": description,
                "environment": answers.get("environment", ""),
                "objects": answers.get("objects", ""),
                "person": answers.get("person", ""),
                "location": answers.get("location", ""),
                "emotion": answers.get("emotion", ""),
                "color": answers.get("color", ""),
            }

            raw_output = description
            print(f"📝 BLIP-2 生成的原始描述: caption={caption} \n description={raw_output}")

            result = json.dumps(structured, ensure_ascii=False)

        # 解析结构化JSON输出
        structured_data = self._parse_structured_output(result)

        grounding_raw = None
        if self._grounding_second_pass and self.model_type == "deepseek-vl":
            g_tokens = min(384, max(128, getattr(self, "_max_new_tokens", 512)))
            grounding_raw = self._deepseek_multimodal_generate(
                image, DEEPSEEK_GROUNDING_PROMPT, max_new_tokens=g_tokens, use_streamer=False
            )
            dets = self._parse_ref_det_output(grounding_raw)
            if dets:
                self._merge_ref_det_into_objects(structured_data, dets)
                print(f"🎯 DeepSeek <|ref|><|det|> 解析到 {len(dets)} 个框: {dets}")
            else:
                print(
                    f"🎯 DeepSeek 定位轮未解析到 <|ref|><|det|>（可换 chat 模型或检查原始输出前 300 字）: "
                    f"{(grounding_raw or '')[:300]!r}"
                )

        self._expand_person_bbox_rel_on_objects(structured_data)

        # 提取标签（用于向后兼容）
        tags = self._extract_tags_from_structured_data(structured_data)

        out = {
            "image_path": image_path,
            "description": raw_output,
            "structured_data": structured_data,
            "tags": tags,
            "model_type": self.model_type,
            "timestamp": datetime.now().isoformat(),
        }
        if grounding_raw is not None:
            out["grounding_raw"] = grounding_raw
        out["inference_time"] = time.time() - start_time
        return out
    
    def _extract_first_json_object(self, text: str) -> str:
        """从文本中提取第一个平衡的大括号 JSON 对象（支持嵌套与字符串内转义）。"""
        start = text.find("{")
        if start < 0:
            return ""
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\" and in_string:
                escape = True
                continue
            if c == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return ""

    def _parse_structured_output(self, result: str) -> Dict:
        """解析模型输出的JSON格式内容"""
        try:
            json_str = self._extract_first_json_object(result)
            if json_str:
                print(f"📝 匹配到的结构化数据: {json_str[:800]}{'...' if len(json_str) > 800 else ''}")
                parsed = json.loads(json_str)
                print(f"📝 提取到结构化数据: {parsed}")
                return parsed
        except (json.JSONDecodeError, Exception):
            import traceback
            print("解析结构化数据时发生异常：")
            traceback.print_exc()

        # 模型输出格式不正确，无法提取结构化数据，返回空字典
        print("模型输出格式不正确，无法提取结构化数据，返回空字典")
        return {}

    def _extract_tags_from_structured_data(self, structured_data: Dict) -> List[str]:
        """从结构化数据中提取标签"""
        tags = []
        seen = set()

        # objects：支持数组 [{"label": "...", "points": [...]}, ...] 或旧版逗号分隔字符串
        obj_val = structured_data.get("objects")
        if isinstance(obj_val, list):
            for item in obj_val:
                if isinstance(item, dict):
                    label = item.get("label") or item.get("name") or ""
                    if isinstance(label, str) and label.strip() and label.strip() not in seen:
                        tags.append(label.strip())
                        seen.add(label.strip())
                elif isinstance(item, str) and item.strip() and item.strip() not in seen:
                    tags.append(item.strip())
                    seen.add(item.strip())
        elif obj_val and isinstance(obj_val, str):
            parts = [p.strip() for p in obj_val.split(",") if p.strip()]
            for part in parts:
                if part not in seen and len(part) > 1:
                    tags.append(part)
                    seen.add(part)

        fields_to_extract = ["scene", "person", "environment", "emotion", "location", "color"]
        for field in fields_to_extract:
            value = structured_data.get(field, "")
            if value and isinstance(value, str):
                parts = [p.strip() for p in value.split(",") if p.strip()]
                for part in parts:
                    if part not in seen and len(part) > 1:
                        tags.append(part)
                        seen.add(part)

        return tags

    def _xy_to_pixel(
        self,
        x: float,
        y: float,
        width: int,
        height: int,
        coord_space: Optional[str] = None,
    ) -> Tuple[int, int]:
        """单点坐标转像素。coord_space: None 自动、'rel'(0~1)、'999'、'1000'。"""
        try:
            xf, yf = float(x), float(y)
        except (TypeError, ValueError):
            return 0, 0
        space = coord_space
        if space is None:
            m = max(xf, yf, 0.0)
            space = "rel" if m <= 1.01 else "1000"
        if space == "rel" or space == "0_1":
            px = int(round(xf * width))
            py = int(round(yf * height))
        elif space == "999":
            px = int(round(xf / 999.0 * width))
            py = int(round(yf / 999.0 * height))
        else:
            px = int(round(xf / 1000.0 * width))
            py = int(round(yf / 1000.0 * height))
        px = max(0, min(width - 1, px))
        py = max(0, min(height - 1, py))
        return px, py

    def _bbox_to_pixel_rect(
        self,
        bbox,
        width: int,
        height: int,
        coord_space: Optional[str] = None,
    ) -> Tuple[int, int, int, int]:
        """bbox [x_min,y_min,x_max,y_max] -> 像素 (left, top, right, bottom)。自动识别 JSON 里常见的 0~1 小数。"""
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return 0, 0, 0, 0
        try:
            x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        except (TypeError, ValueError, IndexError):
            return 0, 0, 0, 0
        space = coord_space
        if space is None:
            m = max(x0, y0, x1, y1)
            space = "rel" if m <= 1.01 else "1000"
        if space == "rel" or space == "0_1":
            px0, py0 = int(round(x0 * width)), int(round(y0 * height))
            px1, py1 = int(round(x1 * width)), int(round(y1 * height))
        elif space == "999":
            x0, x1 = max(0.0, min(999.0, x0)), max(0.0, min(999.0, x1))
            y0, y1 = max(0.0, min(999.0, y0)), max(0.0, min(999.0, y1))
            px0 = int(round(x0 / 999.0 * width))
            py0 = int(round(y0 / 999.0 * height))
            px1 = int(round(x1 / 999.0 * width))
            py1 = int(round(y1 / 999.0 * height))
        else:
            x0, x1 = max(0.0, min(1000.0, x0)), max(0.0, min(1000.0, x1))
            y0, y1 = max(0.0, min(1000.0, y0)), max(0.0, min(1000.0, y1))
            px0, py0 = int(round(x0 / 1000.0 * width)), int(round(y0 / 1000.0 * height))
            px1, py1 = int(round(x1 / 1000.0 * width)), int(round(y1 / 1000.0 * height))
        if px0 > px1:
            px0, px1 = px1, px0
        if py0 > py1:
            py0, py1 = py1, py0
        left, right = min(px0, px1), max(px0, px1)
        top, bottom = min(py0, py1), max(py0, py1)
        if right - left < 2:
            right = min(width - 1, left + 2)
        if bottom - top < 2:
            bottom = min(height - 1, top + 2)
        return left, top, right, bottom

    def _points_axis_aligned_rect(
        self, pairs: List, width: int, height: int, coord_space: Optional[str] = None
    ) -> Tuple[int, int, int, int]:
        xs, ys = [], []
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            try:
                xs.append(float(pair[0]))
                ys.append(float(pair[1]))
            except (TypeError, ValueError):
                continue
        if not xs:
            return 0, 0, 0, 0
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        return self._bbox_to_pixel_rect([x0, y0, x1, y1], width, height, coord_space)

    
    def _extract_model_name(self, model_path: str) -> str:
        """从模型路径中提取模型名称（用于文件名后缀）"""
        if not model_path:
            return "default"

        # 获取路径的最后一部分（目录名）
        model_path = os.path.normpath(model_path)
        base_name = os.path.basename(model_path)

        # 清理模型名称，使其适合作为文件名
        # 移除或替换不安全的字符
        safe_name = re.sub(r'[\\/:*?"<>|]', '_', base_name)

        # 如果名称太长，截取前30个字符
        if len(safe_name) > 30:
            safe_name = safe_name[:30]

        return safe_name if safe_name else "custom"


    def batch_process(self, image_dir: str, output_file: str = "tags.json", 
                        save_sibling: bool = False, save_visualization: bool = True,
                        visualization_dir: str = "image-tag-file"):
        """批量处理目录中的图片
        
        Args:
            image_dir: 图片目录路径
            output_file: 汇总结果输出文件路径
            save_sibling: 是否同时保存每个图片的同级目录JSON文件
            save_visualization: 是否保存标注后的图片到 visualization_dir 目录
            visualization_dir: 标注图片输出目录，默认为 "image-tag-file"
        """
        print(f"📂 批量处理目录: {image_dir}")

        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

        # 收集所有图像文件
        image_files = []
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))

        print(f"找到 {len(image_files)} 张图片")
        
        # 如果启用标注图片，确保输出目录存在
        if save_visualization:
            os.makedirs(visualization_dir, exist_ok=True)
            print(f"标注图片将保存到: {visualization_dir}")

        results = []

        for i, img_path in enumerate(image_files, 1):
            print(f"\n处理进度: {i}/{len(image_files)} - {os.path.basename(img_path)}")

            try:
                result = self.analyze_image(img_path)
                results.append(result)

                # 如果启用，保存到同级目录
                if save_sibling:
                    self.save_result_to_sibling(result)
                
                # 如果启用，保存标注图片
                if save_visualization and 'error' not in result:
                    try:
                        tagged_path = self.create_tag_visualization(
                            img_path,
                            result['tags'],
                            output_dir=visualization_dir,
                            structured_data=result.get('structured_data')
                        )
                        result['tagged_image_path'] = tagged_path
                    except Exception as vis_e:
                        print(f"保存标注图片失败: {vis_e}")

                # 每处理5张图片保存一次汇总结果
                if i % 5 == 0:
                    self._save_results(results, output_file)

            except Exception as e:
                print(f"处理失败 {img_path}: {e}")
                error_result = {
                    "image_path": img_path,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                results.append(error_result)
                # 错误结果也可以保存到同级目录
                if save_sibling:
                    try:
                        self.save_result_to_sibling(error_result)
                    except Exception as save_e:
                        print(f"保存错误结果失败: {save_e}")

        # 最终保存汇总结果
        self._save_results(results, output_file)
        print(f"✅ 批量处理完成！汇总结果保存到: {output_file}")
        if save_sibling:
            print(f"💾 每个图片的独立JSON结果已保存到各自同级目录")
        if save_visualization:
            print(f"🏷️ 标注图片已保存到: {visualization_dir}")

        return results
    
    def _save_results(self, results: List[Dict], output_file: str):
        """保存结果到JSON文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def _get_json_path(self, image_path: str) -> str:
        """根据图片路径生成同目录的JSON文件路径（包含模型名称后缀）"""
        # 获取图片所在目录和文件名（不含扩展名）
        image_dir = os.path.dirname(image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        # 获取模型名称后缀
        model_suffix = getattr(self, 'model_name_suffix', 'default')
        # 组合成JSON路径：同目录下，同名_模型名.json
        json_path = os.path.join(image_dir, f"{base_name}_{model_suffix}.json")
        return json_path

    def save_result_to_sibling(self, result: Dict) -> str:
        """将单个结果保存到与图片同级的JSON文件"""
        image_path = result.get("image_path")
        if not image_path:
            raise ValueError("结果中缺少 image_path 字段")

        json_path = self._get_json_path(image_path)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"💾 结果已保存到: {json_path}")
        return json_path
    
    def create_tag_visualization(self, image_path: str, tags: List[str], output_path: str = None, 
                                  output_dir: str = "image-tag-file", structured_data: Dict = None) -> str:
        """创建带有标签可视化的图片
        
        Args:
            image_path: 原始图片路径
            tags: 标签列表
            output_path: 自定义输出路径（如果为None，则根据规则自动生成）
            output_dir: 输出目录，默认为 "image-tag-file"
            structured_data: 结构化数据，用于在图片上显示更多信息
        """
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        
        # 尝试加载支持中文的字体
        font = None
        font_bold = None
        # Windows 系统常见中文字体路径
        chinese_fonts = [
            "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
            "C:/Windows/Fonts/msyhbd.ttc",    # 微软雅黑粗体
            "C:/Windows/Fonts/simhei.ttf",    # 黑体
            "C:/Windows/Fonts/simsun.ttc",    # 宋体
            "C:/Windows/Fonts/simkai.ttf",    # 楷体
        ]
        
        for font_path in chinese_fonts:
            try:
                font = ImageFont.truetype(font_path, 20)
                font_bold = ImageFont.truetype(font_path, 22)
                break
            except:
                continue
        
        # 如果没有找到中文字体，尝试默认字体
        if font is None:
            try:
                font = ImageFont.truetype("arial.ttf", 20)
                font_bold = ImageFont.truetype("arial.ttf", 22)
            except:
                font = ImageFont.load_default()
                font_bold = font

        # 在图像上绘制 objects：优先 bbox，否则 points（多点用轴对齐外接矩形，更清晰）
        objects_raw = (structured_data or {}).get("objects") if structured_data else None
        if isinstance(objects_raw, list) and objects_raw:
            palette = [
                (255, 64, 64), (64, 220, 64), (64, 128, 255), (255, 200, 64),
                (200, 64, 255), (64, 255, 255), (255, 128, 192),
            ]
            iw, ih = image.size
            for oi, item in enumerate(objects_raw):
                if not isinstance(item, dict):
                    continue
                label = item.get("label") or item.get("name") or ""
                color = palette[oi % len(palette)]
                bbox = item.get("bbox") or item.get("xyxy")
                pts = item.get("points") if isinstance(item.get("points"), list) else None
                cspace = item.get("coord_space")

                rect = None
                if bbox is not None:
                    rect = self._bbox_to_pixel_rect(bbox, iw, ih, cspace)
                    left, top, right, bottom = rect
                    if right > left and bottom > top:
                        draw.rectangle([left, top, right, bottom], outline=color, width=4)
                elif pts:
                    pixel_pts: List[Tuple[int, int]] = []
                    for pair in pts:
                        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                            continue
                        px, py = self._xy_to_pixel(pair[0], pair[1], iw, ih, cspace)
                        pixel_pts.append((px, py))
                    if len(pixel_pts) >= 2:
                        rect = self._points_axis_aligned_rect(pts, iw, ih, cspace)
                        l, t, r, b = rect
                        if r > l and b > t:
                            draw.rectangle([l, t, r, b], outline=color, width=4)
                        for (px, py) in pixel_pts:
                            rad = 6
                            draw.ellipse(
                                [px - rad, py - rad, px + rad, py + rad],
                                outline=color,
                                width=2,
                            )
                    elif len(pixel_pts) == 1:
                        px, py = pixel_pts[0]
                        rad = 8
                        draw.ellipse([px - rad, py - rad, px + rad, py + rad], outline=color, width=3)

                if label:
                    if rect:
                        l, t, r, b = rect
                        tx, ty = l + 4, t - 6
                    elif pts and isinstance(pts, list):
                        first = pts[0]
                        if isinstance(first, (list, tuple)) and len(first) >= 2:
                            tx, ty = self._xy_to_pixel(first[0], first[1], iw, ih, cspace)
                            tx, ty = tx + 10, ty - 10
                        else:
                            tx, ty = 10, 30
                    else:
                        continue
                    tx = max(2, min(iw - 160, tx))
                    ty = max(2, min(ih - 26, ty))
                    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        draw.text((tx + dx, ty + dy), str(label), fill=(0, 0, 0), font=font)
                    draw.text((tx, ty), str(label), fill=color, font=font)
        
        # 在图片上添加标签区域背景
        y_offset = 10
        label_width = 280
        line_height = 32
        
        # 绘制标签列表
        for i, tag in enumerate(tags):
            draw.rectangle([10, y_offset, 10 + label_width, y_offset + line_height], 
                          fill=(0, 0, 0, 180))
            draw.text((15, y_offset + 5), f"{i+1}. {tag}", fill=(255, 255, 255), font=font)
            y_offset += line_height + 3
        
        # 如果有结构化数据，在右侧添加更多信息
        if structured_data:
            x_right = image.width - 290
            y_right = 10
            right_width = 280
            
            # 绘制背景
            info_lines = []
            if structured_data.get("scene"):
                info_lines.append(("场景", structured_data["scene"]))
            if structured_data.get("person"):
                info_lines.append(("人物", structured_data["person"]))
            if structured_data.get("environment"):
                info_lines.append(("环境", structured_data["environment"]))
            if structured_data.get("emotion"):
                info_lines.append(("氛围", structured_data["emotion"]))
            if structured_data.get("color"):
                info_lines.append(("颜色", structured_data["color"]))
            
            if info_lines:
                total_height = len(info_lines) * (line_height + 3) + 10
                draw.rectangle([x_right, y_right, x_right + right_width, y_right + total_height],
                              fill=(0, 0, 0, 180))
                
                for label, value in info_lines[:6]:
                    draw.text((x_right + 5, y_right + 5), f"{label}: {value}", 
                             fill=(255, 255, 255), font=font)
                    y_right += line_height + 3
        
        # 确定输出路径
        if output_path is None:
            # 获取原文件扩展名
            orig_ext = os.path.splitext(image_path)[1].lower()
            if orig_ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                orig_ext = '.jpg'
            
            # 获取与 JSON 文件一致的文件名（不含扩展名）
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            model_suffix = getattr(self, 'model_name_suffix', 'default')
            new_filename = f"{base_name}_{model_suffix}{orig_ext}"
            
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, new_filename)
        
        image.save(output_path, quality=95)
        print(f"可视化图片保存到: {output_path}")
        
        return output_path

# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek/BLIP 图像打标")
    parser.add_argument("--model_path", type=str, default=None,
                        help="本地模型目录（推荐 chat 权重，如 deepseek-vl-7b-chat）；不传则使用 HF deepseek-vl-7b-chat")
    parser.add_argument("--model_type", type=str, default="deepseek-vl", choices=["deepseek-vl", "blip2"],
                        help="模型类型：deepseek-vl 或 blip2，默认 deepseek-vl")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"],
                        help="设备：cuda 或 cpu，不传则自动检测（有显卡会用 GPU，快很多）")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="生成描述最大 token 数，默认 512（JSON输出需要更多token）")
    parser.add_argument("--image", type=str, default="test.png", help="单张测试图片路径")
    parser.add_argument("--batch_dir", type=str, default=None, help="批量处理图片目录")
    parser.add_argument("--output", type=str, default="tags.json", help="批量结果输出文件")
    parser.add_argument("--save_sibling", action="store_true",
                        help="批量处理时，同时保存每个图片的独立结果到同级目录（如 aa/test1.jpg -> aa/test1.json）")
    parser.add_argument("--no_visualization", action="store_true",
                        help="不生成标注后的可视化图片（默认会生成）")
    parser.add_argument("--visualization_dir", type=str, default="image-tag-file",
                        help="标注图片输出目录，默认为 image-tag-file")
    parser.add_argument(
        "--no_grounding_2nd",
        action="store_true",
        help="DeepSeek-VL 禁用第二路 <|ref|><|det|> 定位（只做 JSON 描述，略快）",
    )
    parser.add_argument(
        "--no_expand_person_bbox",
        action="store_true",
        help="禁用对人物 0～1 相对 bbox 的适度外扩启发式",
    )
    args = parser.parse_args()

    # 初始化打标器（传入命令行 model_path、model_type、device、max_tokens）
    tagger = DeepSeekImageTagger(
        model_path=args.model_path,
        model_type=args.model_type,
        device=args.device,
        max_new_tokens=args.max_tokens,
        grounding_second_pass=not args.no_grounding_2nd,
        expand_person_bbox=not args.no_expand_person_bbox,
    )

    # 批量处理 或 单张测试
    if args.batch_dir and os.path.isdir(args.batch_dir):
        tagger.batch_process(args.batch_dir, output_file=args.output, 
                             save_sibling=args.save_sibling,
                             save_visualization=not args.no_visualization,
                             visualization_dir=args.visualization_dir)
    else:
        test_image = args.image
        if os.path.exists(test_image):
            print("\n🔍 分析单张图片...")
            result = tagger.analyze_image(test_image)
            print(f"\n📝 原始输出: {result['description']}")
            print(f"\n📊 结构化数据:")
            structured = result.get('structured_data', {})
            for key, value in structured.items():
                print(f"  {key}: {value}")
            print(f"\n🏷️ 标签: {', '.join(result['tags'])}")
            print(f"⏱️ 推理时间: {result['inference_time']:.2f}秒")
            # 保存标注图片到 image-tag-file 目录
            tagged_image_path = tagger.create_tag_visualization(
                test_image, 
                result['tags'], 
                structured_data=result.get('structured_data')
            )
            print(f"🏷️ 标注图片: {tagged_image_path}")
            # 保存 JSON 到与图片同级的目录
            tagger.save_result_to_sibling(result)
        else:
            print(f"测试图片不存在: {test_image}")
            print("请准备一张测试图片，或使用 --batch_dir 指定目录批量处理")