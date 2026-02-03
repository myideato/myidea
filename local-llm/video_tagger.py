import cv2
import os
import json
from PIL import Image
import numpy as np
from datetime import datetime
from image_tagger import DeepSeekImageTagger  # 重用图片打标器

class VideoFrameTagger:
    def __init__(self, model_path=None):
        """初始化视频打标系统"""
        print("🎥 初始化视频帧打标系统...")
        self.image_tagger = DeepSeekImageTagger(model_path)
        self.frame_interval = 10  # 每10帧采样一帧
        self.max_frames = 100  # 最大处理帧数
    
    def extract_keyframes(self, video_path: str, output_dir: str = "frames"):
        """从视频中提取关键帧"""
        print(f"提取关键帧: {video_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return []
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"视频信息: {total_frames}帧, {fps:.1f}FPS, {duration:.1f}秒")
        
        frame_paths = []
        frame_count = 0
        saved_count = 0
        
        while saved_count < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 按间隔采样
            if frame_count % self.frame_interval == 0:
                # 转换为RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # 保存帧
                frame_filename = f"frame_{saved_count:04d}_t{frame_count//fps:.1f}s.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                pil_image.save(frame_path, quality=85)
                
                frame_paths.append({
                    "path": frame_path,
                    "frame_index": frame_count,
                    "timestamp": frame_count / fps
                })
                
                saved_count += 1
                print(f"  保存帧: {frame_filename}")
            
            frame_count += 1
        
        cap.release()
        print(f"✅ 提取了 {saved_count} 个关键帧")
        
        return frame_paths
    
    def analyze_video(self, video_path: str, output_file: str = "video_tags.json"):
        """分析视频并生成标签"""
        print(f"\n🎬 开始分析视频: {video_path}")
        
        # 1. 提取关键帧
        frames_dir = "extracted_frames"
        frames = self.extract_keyframes(video_path, frames_dir)
        
        if not frames:
            print("❌ 未能提取到帧")
            return None
        
        # 2. 分析每一帧
        results = []
        
        for i, frame_info in enumerate(frames):
            print(f"\n分析帧 {i+1}/{len(frames)}: {os.path.basename(frame_info['path'])}")
            
            try:
                # 分析单帧
                frame_result = self.image_tagger.analyze_image(frame_info['path'])
                
                # 添加时间信息
                frame_result.update({
                    "video_path": video_path,
                    "frame_index": frame_info["frame_index"],
                    "timestamp": frame_info["timestamp"],
                    "frame_time": f"{frame_info['timestamp']:.1f}s"
                })
                
                results.append(frame_result)
                
                # 显示进度
                print(f"  标签: {', '.join(frame_result['tags'][:5])}...")
                
            except Exception as e:
                print(f"  分析失败: {e}")
                results.append({
                    "video_path": video_path,
                    "frame_path": frame_info['path'],
                    "frame_index": frame_info["frame_index"],
                    "timestamp": frame_info["timestamp"],
                    "error": str(e)
                })
        
        # 3. 聚合视频级标签
        video_tags = self._aggregate_video_tags(results)
        
        # 4. 创建视频摘要
        video_summary = {
            "video_path": video_path,
            "total_frames_analyzed": len(results),
            "analysis_time": datetime.now().isoformat(),
            "video_tags": video_tags,
            "frame_analysis": results,
            "summary": self._generate_video_summary(results)
        }
        
        # 5. 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(video_summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 视频分析完成！结果保存到: {output_file}")
        print(f"🎯 视频级标签: {', '.join(video_tags[:10])}...")
        
        return video_summary
    
    def _aggregate_video_tags(self, frame_results: list) -> list:
        """聚合所有帧的标签，生成视频级标签"""
        tag_freq = {}
        
        for result in frame_results:
            if "tags" in result:
                for tag in result["tags"]:
                    tag_freq[tag] = tag_freq.get(tag, 0) + 1
        
        # 按频率排序
        sorted_tags = sorted(tag_freq.items(), key=lambda x: x[1], reverse=True)
        
        # 返回频率最高的20个标签
        return [tag for tag, freq in sorted_tags[:20]]
    
    def _generate_video_summary(self, frame_results: list) -> str:
        """生成视频内容摘要"""
        # 收集所有描述
        descriptions = []
        for result in frame_results:
            if "description" in result:
                descriptions.append(result["description"])
        
        # 简单摘要：取前几个描述的合并
        if descriptions:
            summary = "视频主要内容：\n"
            for i, desc in enumerate(descriptions[:5]):
                summary += f"  {i+1}. {desc[:100]}...\n"
            return summary
        else:
            return "未能生成视频摘要"
    
    def create_video_report(self, video_path: str, output_html: str = "video_report.html"):
        """生成HTML报告"""
        print(f"生成视频分析报告: {output_html}")
        
        # 读取分析结果
        json_file = "video_tags.json"
        if not os.path.exists(json_file):
            print(f"请先运行视频分析: {video_path}")
            return
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 生成HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>视频分析报告 - {os.path.basename(video_path)}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .tags {{ margin: 20px 0; }}
                .tag {{ display: inline-block; background: #4CAF50; color: white; 
                        padding: 5px 10px; margin: 5px; border-radius: 3px; }}
                .frame {{ margin: 20px 0; border: 1px solid #ddd; padding: 10px; }}
                .frame-img {{ max-width: 300px; }}
                .timestamp {{ color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎬 视频分析报告</h1>
                <p><strong>视频文件:</strong> {data['video_path']}</p>
                <p><strong>分析时间:</strong> {data['analysis_time']}</p>
                <p><strong>分析帧数:</strong> {data['total_frames_analyzed']}</p>
            </div>
            
            <div class="summary">
                <h2>📋 视频摘要</h2>
                <pre>{data['summary']}</pre>
            </div>
            
            <div class="tags">
                <h2>🏷️ 视频标签</h2>
                {"".join([f'<span class="tag">{tag}</span>' for tag in data['video_tags'][:20]])}
            </div>
            
            <div class="frames">
                <h2>🎞️ 关键帧分析</h2>
        """
        
        for i, frame in enumerate(data['frame_analysis'][:20]):  # 最多显示20帧
            if 'frame_path' in frame and os.path.exists(frame['frame_path']):
                html_content += f"""
                <div class="frame">
                    <h3>帧 #{i+1} - {frame.get('frame_time', 'N/A')}</h3>
                    <img class="frame-img" src="{frame['frame_path']}" alt="Frame {i+1}">
                    <p class="timestamp">时间戳: {frame.get('timestamp', 'N/A')}秒</p>
                    <p><strong>描述:</strong> {frame.get('description', 'N/A')[:200]}...</p>
                    <p><strong>标签:</strong> {', '.join(frame.get('tags', []))[:10]}</p>
                </div>
                """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        with open(output_html, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML报告已生成: {output_html}")
        return output_html

# 使用示例
if __name__ == "__main__":
    # 初始化视频打标器
    video_tagger = VideoFrameTagger()
    
    # 测试视频文件
    test_video = "test.mp4"  # 替换为你的视频路径
    
    if os.path.exists(test_video):
        print("开始视频分析...")
        
        # 分析视频
        result = video_tagger.analyze_video(test_video, "video_analysis.json")
        
        # 生成报告
        video_tagger.create_video_report(test_video, "video_report.html")
        
        print("\n📊 分析完成！")
        print("1. 查看详细结果: video_analysis.json")
        print("2. 查看HTML报告: video_report.html")
        print("3. 查看提取的帧: extracted_frames/")
        
    else:
        print(f"测试视频不存在: {test_video}")
        print("请准备一个测试视频（MP4格式）")