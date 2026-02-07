import torch
from PIL import Image, ImageDraw, ImageFont
import os
import argparse
from typing import List, Dict, Tuple
import json
import time
from datetime import datetime

class DeepSeekImageTagger:
    def __init__(self, model_path=None, device=None, max_new_tokens=256):
        """初始化DeepSeek-VL打标系统。device 可选 'cuda'/'cpu'，不传则自动检测。"""
        print("🚀 初始化DeepSeek图像打标系统...")
        self._max_new_tokens = max_new_tokens
        
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
        
        try:
            # 尝试加载 DeepSeek-VL（含 Hybrid），与 download_model 一致：AutoModelForImageTextToText + AutoProcessor
            from transformers import AutoModelForImageTextToText, AutoProcessor

            default_hf = "deepseek-community/deepseek-vl-7b-base"
            if model_path:
                model_path_abs = os.path.abspath(os.path.normpath(model_path))
                if os.path.isdir(model_path_abs):
                    model_name = model_path_abs
                else:
                    model_name = default_hf
            else:
                model_name = default_hf

            print(f"加载模型: {model_name}")

            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            device_map = "cuda" if self.device == "cuda" else "cpu"
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                device_map=device_map,
            )
            self.model.eval()

            self.model_type = "deepseek-vl"
            
        except Exception as e:
            print(f"DeepSeek-VL加载失败: {e}")
            print("使用备用模型: BLIP-2")
            self._load_blip_backup()
    
    def _load_blip_backup(self):
        """加载备用BLIP-2模型"""
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        self.model.eval()
        self.model_type = "blip2"
    
    def analyze_image(self, image_path: str, prompt: str = None) -> Dict:
        """分析图像并生成描述和标签"""
        print(f"分析图像: {image_path}")
        
        # 打开图像
        image = Image.open(image_path).convert("RGB")
        
        start_time = time.time()
        
        if self.model_type == "deepseek-vl":
            # DeepSeek-VL / DeepseekVLHybrid：使用 chat 模板 + apply_chat_template
            if prompt is None:
                prompt = "详细描述这张图片的内容，列出主要物体、场景、颜色、动作等要素。"
            # content 支持 "image"(PIL) 或 "url"(本地用 file://)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            try:
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    padding=True,
                    truncation=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            except (TypeError, KeyError):
                # 部分 processor 只认 url，用 file:// 本地路径
                path_url = "file:///" + os.path.abspath(image_path).replace("\\", "/")
                messages[0]["content"][0] = {"type": "image", "url": path_url}
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    padding=True,
                    truncation=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            # 移到模型所在 device（BatchFeature 支持 .to()）
            device = getattr(self.model, "device", torch.device("cpu"))
            dtype = getattr(self.model, "dtype", torch.float32)
            if hasattr(dtype, "dtype"):
                dtype = dtype.dtype
            if hasattr(inputs, "to"):
                inputs = inputs.to(device, dtype=dtype)
            else:
                inputs = {k: (v.to(device, dtype=dtype) if hasattr(v, "to") else v) for k, v in inputs.items()}
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=getattr(self, "_max_new_tokens", 256), do_sample=True, temperature=0.7, top_p=0.9)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            decoded = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            result = decoded[0] if decoded else ""
            
        else:  # BLIP-2
            if prompt is None:
                prompt = "Question: 详细描述这张图片的内容。 Answer:"
            
            inputs = self.processor(image, prompt, return_tensors="pt").to(self.device, torch.float32)
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=200)
            
            result = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        inference_time = time.time() - start_time
        
        # 提取标签
        tags = self._extract_tags_from_description(result)
        
        return {
            "image_path": image_path,
            "description": result,
            "tags": tags,
            "inference_time": inference_time,
            "model_type": self.model_type,
            "timestamp": datetime.now().isoformat()
        }
    
    def _extract_tags_from_description(self, description: str) -> List[str]:
        """从描述中提取关键词作为标签"""
        # 简单关键词提取（可根据需求扩展）
        import re
        
        # 移除常见停用词
        stop_words = {"的", "了", "在", "和", "是", "有", "这个", "一个", "图片", "图像"}
        
        # 提取名词性短语
        words = re.findall(r'[\u4e00-\u9fff\w]+', description.lower())
        
        # 过滤和去重
        tags = []
        seen = set()
        
        for word in words:
            if (len(word) > 1 and 
                word not in stop_words and 
                word not in seen and
                not word.isdigit()):
                tags.append(word)
                seen.add(word)
        
        # 返回前20个标签
        return tags[:20]
    
    def batch_process(self, image_dir: str, output_file: str = "tags.json"):
        """批量处理目录中的图片"""
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
        
        results = []
        
        for i, img_path in enumerate(image_files, 1):
            print(f"\n处理进度: {i}/{len(image_files)} - {os.path.basename(img_path)}")
            
            try:
                result = self.analyze_image(img_path)
                results.append(result)
                
                # 每处理5张图片保存一次
                if i % 5 == 0:
                    self._save_results(results, output_file)
                    
            except Exception as e:
                print(f"处理失败 {img_path}: {e}")
                results.append({
                    "image_path": img_path,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        # 最终保存
        self._save_results(results, output_file)
        print(f"✅ 批量处理完成！结果保存到: {output_file}")
        
        return results
    
    def _save_results(self, results: List[Dict], output_file: str):
        """保存结果到JSON文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    def create_tag_visualization(self, image_path: str, tags: List[str], output_path: str = None):
        """创建带有标签可视化的图片"""
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        
        # 尝试加载字体
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # 在图片上添加标签
        y_offset = 10
        for i, tag in enumerate(tags[:10]):  # 只显示前10个标签
            draw.rectangle([10, y_offset, 200, y_offset + 30], fill=(0, 0, 0, 128))
            draw.text((15, y_offset + 5), f"{i+1}. {tag}", fill=(255, 255, 255), font=font)
            y_offset += 35
        
        if output_path is None:
            output_path = f"tagged_{os.path.basename(image_path)}"
        
        image.save(output_path)
        print(f"可视化图片保存到: {output_path}")
        
        return output_path

# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek/BLIP 图像打标")
    parser.add_argument("--model_path", type=str, default=None,
                        help="本地 DeepSeek-VL 模型目录，如 .\\deepseek_vl_model\\")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"],
                        help="设备：cuda 或 cpu，不传则自动检测（有显卡会用 GPU，快很多）")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="生成描述最大 token 数，越小越快，默认 256")
    parser.add_argument("--image", type=str, default="test.png", help="单张测试图片路径")
    parser.add_argument("--batch_dir", type=str, default=None, help="批量处理图片目录")
    parser.add_argument("--output", type=str, default="tags.json", help="批量结果输出文件")
    args = parser.parse_args()

    # 初始化打标器（传入命令行 model_path、device、max_tokens）
    tagger = DeepSeekImageTagger(model_path=args.model_path, device=args.device, max_new_tokens=args.max_tokens)

    # 批量处理 或 单张测试
    if args.batch_dir and os.path.isdir(args.batch_dir):
        tagger.batch_process(args.batch_dir, output_file=args.output)
    else:
        test_image = args.image
        if os.path.exists(test_image):
            print("\n🔍 分析单张图片...")
            result = tagger.analyze_image(test_image)
            print(f"\n📝 描述: {result['description']}")
            print(f"\n🏷️ 标签: {', '.join(result['tags'])}")
            print(f"⏱️ 推理时间: {result['inference_time']:.2f}秒")
            tagger.create_tag_visualization(test_image, result['tags'])
            with open("single_result.json", "w", encoding="utf-8") as f:
                json.dump([result], f, ensure_ascii=False, indent=2)
        else:
            print(f"测试图片不存在: {test_image}")
            print("请准备一张测试图片，或使用 --batch_dir 指定目录批量处理")