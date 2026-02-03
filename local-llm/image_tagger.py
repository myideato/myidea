import torch
from PIL import Image, ImageDraw, ImageFont
import os
from typing import List, Dict, Tuple
import json
import time
from datetime import datetime

class DeepSeekImageTagger:
    def __init__(self, model_path=None):
        """初始化DeepSeek-VL打标系统"""
        print("🚀 初始化DeepSeek图像打标系统...")
        
        self.device = "cpu"
        print(f"使用设备: {self.device}")
        
        try:
            # 尝试加载DeepSeek-VL
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor
            
            if model_path and os.path.exists(model_path):
                model_name = model_path
            else:
                model_name = "deepseek-ai/deepseek-vl-7b"
            
            print(f"加载模型: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            self.image_processor = AutoImageProcessor.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            
            # 加载模型（CPU模式）
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                device_map="cpu"
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
            # DeepSeek-VL处理流程
            if prompt is None:
                prompt = "详细描述这张图片的内容，列出主要物体、场景、颜色、动作等要素。"
            
            # 准备输入
            inputs = self.image_processor(images=image, return_tensors="pt")
            
            # 编码文本
            text_encoding = self.tokenizer(
                prompt, 
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # 合并输入
            inputs.update(text_encoding)
            
            # 推理
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            
            # 解码结果
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
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
    # 初始化打标器
    tagger = DeepSeekImageTagger()
    
    # 测试单张图片
    test_image = "test.jpg"  # 替换为你的图片路径
    
    if os.path.exists(test_image):
        print("\n🔍 分析单张图片...")
        result = tagger.analyze_image(test_image)
        
        print(f"\n📝 描述: {result['description']}")
        print(f"\n🏷️ 标签: {', '.join(result['tags'])}")
        print(f"⏱️ 推理时间: {result['inference_time']:.2f}秒")
        
        # 创建可视化
        tagger.create_tag_visualization(test_image, result['tags'])
        
        # 保存结果
        with open("single_result.json", "w", encoding="utf-8") as f:
            json.dump([result], f, ensure_ascii=False, indent=2)
    else:
        print(f"测试图片不存在: {test_image}")
        print("请准备一张测试图片，或运行批量处理")
    
    # 批量处理示例（取消注释以使用）
    # tagger.batch_process("./images", "batch_tags.json")