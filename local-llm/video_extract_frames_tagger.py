import argparse
import cv2
import os
import json
from PIL import Image
from datetime import datetime
from image_tagger import DeepSeekImageTagger  # 重用图片打标器

class VideoFrameTagger:
    def __init__(
        self,
        model_path=None,
        model_type="deepseek-vl",
        device=None,
        max_new_tokens=256,
        frame_interval=10,
        max_frames=100,
    ):
        """初始化视频打标系统"""
        print("🎥 初始化视频帧打标系统...")
        self.image_tagger = DeepSeekImageTagger(
            model_path=model_path,
            model_type=model_type,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        self.frame_interval = frame_interval  # 每隔多少帧采样一帧
        self.max_frames = max_frames  # 最大处理帧数
    
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
    
    def analyze_video(
        self,
        video_path: str,
        output_file: str = "video_tags.json",
        frames_dir: str = "extracted_frames",
    ):
        """分析视频并生成标签"""
        print(f"\n🎬 开始分析视频: {video_path}")
        
        # 1. 提取关键帧
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
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="视频抽帧 + DeepSeek-VL/BLIP-2 逐帧图像打标（与 image_tagger 参数风格一致）"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="本地模型目录，如 .\\deepseek_vl_model\\ 或 .\\blip2_model\\",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="deepseek-vl",
        choices=["deepseek-vl", "blip2"],
        help="模型类型：deepseek-vl 或 blip2，默认 deepseek-vl",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="设备：cuda 或 cpu; 不传则自动检测",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="每帧描述最大 token 数，默认 512（与 image_tagger 一致）",
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="输入视频路径（MP4 等 OpenCV 可读格式）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="汇总结果 JSON；未指定时为「视频同级目录 / 视频名_模型名.json」（模型名规则同 image_tagger）",
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        default=None,
        help="抽帧保存目录；未指定时为「视频同级目录 / 视频名_模型名_extracted_frames」",
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=10,
        help="每隔多少帧保存一帧，默认 10",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=100,
        help="最多分析多少帧，默认 100",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"视频不存在或不是文件: {args.video}")
        raise SystemExit(1)

    video_tagger = VideoFrameTagger(
        model_path=args.model_path,
        model_type=args.model_type,
        device=args.device,
        max_new_tokens=args.max_tokens,
        frame_interval=args.frame_interval,
        max_frames=args.max_frames,
    )

    video_abs = os.path.abspath(os.path.normpath(args.video))
    video_dir = os.path.dirname(video_abs)
    video_stem = os.path.splitext(os.path.basename(video_abs))[0]
    model_suffix = getattr(
        video_tagger.image_tagger, "model_name_suffix", "default"
    )

    output_file = args.output
    if output_file is None:
        output_file = os.path.join(video_dir, f"{video_stem}_{model_suffix}.json")

    frames_dir = args.frames_dir
    if frames_dir is None:
        frames_dir = os.path.join(
            video_dir, f"{video_stem}_{model_suffix}_extracted_frames"
        )

    print(f"汇总 JSON: {output_file}")
    print(f"抽帧目录: {frames_dir}")

    video_tagger.analyze_video(
        args.video,
        output_file=output_file,
        frames_dir=frames_dir,
    )