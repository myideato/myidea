import argparse
import json
import os
from datetime import datetime
from io import BytesIO

import numpy as np
import open3d as o3d
from PIL import Image

from image_tagger import DeepSeekImageTagger


def _orbit_orthographic_uv(
    pts_xyz: np.ndarray,
    center: np.ndarray,
    azim_deg: float,
    elev_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """世界系 Y 向上：相机在球坐标 (azimuth, elevation) 方向看向 center，正交投影到画布的 (u,v)。"""
    phi = np.radians(elev_deg)
    theta = np.radians(azim_deg)
    ex = np.cos(phi) * np.sin(theta)
    ey = np.sin(phi)
    ez = np.cos(phi) * np.cos(theta)
    eye_dir = np.array([ex, ey, ez], dtype=np.float64)
    n = float(np.linalg.norm(eye_dir))
    if n < 1e-12:
        eye_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        eye_dir /= n
    view_into = -eye_dir
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(world_up, view_into)
    rn = float(np.linalg.norm(right))
    if rn < 1e-8:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        right /= rn
    up_cam = np.cross(view_into, right)
    up_cam /= float(np.linalg.norm(up_cam))
    p = pts_xyz - center
    u = p @ right
    v = p @ up_cam
    return u, v


def _ply_to_points_and_colors(ply_path: str) -> tuple[np.ndarray, np.ndarray] | None:
    """从 PLY 得到 Nx3 坐标与 Nx3 RGB（0–1）。含三角面时优先读网格；无顶点色则用法线→伪彩色。"""
    mesh = o3d.io.read_triangle_mesh(ply_path)
    if not mesh.is_empty() and len(mesh.triangles) > 0:
        pts = np.asarray(mesh.vertices, dtype=np.float64)
        if mesh.has_vertex_colors():
            cols = np.asarray(mesh.vertex_colors, dtype=np.float64)
            print("已加载网格顶点颜色（RGB）")
        else:
            mesh.compute_vertex_normals()
            n = np.asarray(mesh.vertex_normals, dtype=np.float64)
            cols = np.clip((n + 1.0) * 0.5, 0.0, 1.0)
            print(
                "该 PLY 网格顶点无 RGB 属性，已用法线生成伪彩色（与 Open3D 带光照预览不同；"
                "真彩色需文件中含 red/green/blue 或单独纹理）。"
            )
    else:
        pcd = o3d.io.read_point_cloud(ply_path)
        if len(pcd.points) == 0:
            return None
        pts = np.asarray(pcd.points, dtype=np.float64)
        if pcd.has_colors():
            cols = np.asarray(pcd.colors, dtype=np.float64)
            print("已加载点云顶点颜色（RGB）")
        else:
            pcd.estimate_normals()
            n = np.asarray(pcd.normals, dtype=np.float64)
            cols = np.clip((n + 1.0) * 0.5, 0.0, 1.0)
            print("点云无颜色；已估计法线并生成伪彩色。")
    if cols.size and float(cols.max()) > 1.0 + 1e-6:
        cols = cols / 255.0
    cols = np.clip(cols, 0.0, 1.0)
    return pts, cols


class PlyOrbitFrameTagger:
    """点云 PLY：绕世界 Y 轴均匀视角渲染为「帧」，再复用 VL 图像打标。"""

    def __init__(
        self,
        model_path=None,
        model_type="deepseek-vl",
        device=None,
        max_new_tokens=256,
        num_views=12,
        max_points=120_000,
        render_width=960,
        render_height=720,
        view_elev=18.0,
    ):
        print("📦 初始化点云环绕视角打标系统...")
        self.image_tagger = DeepSeekImageTagger(
            model_path=model_path,
            model_type=model_type,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        self.num_views = num_views
        self.max_points = max_points
        self.render_width = render_width
        self.render_height = render_height
        self.view_elev = view_elev

    def extract_orbit_frames(self, ply_path: str, output_dir: str) -> list:
        """绕 Y 轴均匀 azimuth + view_elev 仰角，用 2D 正交投影出图（避免 mplot3D 上色不稳定）。"""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "环绕渲染需要 matplotlib，请执行: pip install matplotlib"
            ) from e

        print(f"读取: {ply_path}")
        loaded = _ply_to_points_and_colors(ply_path)
        if loaded is None:
            print("无法从 PLY 得到几何数据")
            return []
        pts, c_rgb = loaded

        n = pts.shape[0]
        if n > self.max_points:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, size=self.max_points, replace=False)
            pts = pts[idx]
            c_rgb = c_rgb[idx]
            print(f"点数为 {n}，随机采样至 {self.max_points} 用于渲染")

        center = np.mean(pts, axis=0)
        npts = pts.shape[0]

        os.makedirs(output_dir, exist_ok=True)
        dpi = 100
        fig_w = self.render_width / dpi
        fig_h = self.render_height / dpi

        frame_paths = []
        for i in range(self.num_views):
            azim = 360.0 * i / self.num_views
            u, v = _orbit_orthographic_uv(pts, center, azim, self.view_elev)
            fig, ax = plt.subplots(
                figsize=(fig_w, fig_h),
                dpi=dpi,
                facecolor="white",
            )
            ax.set_facecolor("white")
            # 2D scatter：c=(N,3) 为 RGB，比 Axes3D 可靠得多
            pt_size = float(max(0.25, min(8.0, 12_000.0 / max(npts, 1) ** 0.45)))
            ax.scatter(
                u,
                v,
                s=pt_size,
                c=c_rgb,
                linewidths=0,
                edgecolors="none",
                rasterized=True,
            )
            um, vm = float(np.mean(u)), float(np.mean(v))
            span_u = float(np.max(u) - np.min(u))
            span_v = float(np.max(v) - np.min(v))
            span_xy = max(span_u, span_v, 1e-9)
            half = 0.55 * span_xy
            ax.set_xlim(um - half, um + half)
            ax.set_ylim(vm - half, vm + half)
            ax.set_aspect("equal")
            ax.axis("off")
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

            yaw_rad = float(np.radians(azim))
            frame_filename = f"orbit_{i:04d}_azim{azim:.1f}deg.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            fig.patch.set_facecolor("white")
            buf = BytesIO()
            fig.savefig(
                buf,
                format="png",
                dpi=dpi,
                facecolor="white",
                bbox_inches="tight",
                pad_inches=0.02,
            )
            plt.close(fig)
            buf.seek(0)
            Image.open(buf).convert("RGB").save(frame_path, quality=88, format="JPEG")

            frame_paths.append(
                {
                    "path": os.path.abspath(frame_path),
                    "frame_index": i,
                    "azimuth_deg": azim,
                    "yaw_rad": yaw_rad,
                }
            )
            print(f"  保存视角帧: {frame_filename}")

        print(f"✅ 共渲染 {len(frame_paths)} 个环绕视角")
        return frame_paths

    def analyze_ply(
        self,
        ply_path: str,
        output_file: str = "ply_orbit_tags.json",
        frames_dir: str = "ply_orbit_frames",
    ):
        print(f"\n🧊 开始分析点云: {ply_path}")

        frames = self.extract_orbit_frames(ply_path, frames_dir)
        if not frames:
            print("❌ 未能生成视角图")
            return None

        results = []
        for i, frame_info in enumerate(frames):
            print(f"\n分析视角 {i + 1}/{len(frames)}: {os.path.basename(frame_info['path'])}")
            try:
                frame_result = self.image_tagger.analyze_image(frame_info["path"])
                frame_result.update(
                    {
                        "ply_path": ply_path,
                        "frame_index": frame_info["frame_index"],
                        "azimuth_deg": frame_info["azimuth_deg"],
                        "yaw_rad": frame_info["yaw_rad"],
                    }
                )
                results.append(frame_result)
                print(f"  标签: {', '.join(frame_result['tags'][:5])}...")
            except Exception as e:
                print(f"  分析失败: {e}")
                results.append(
                    {
                        "ply_path": ply_path,
                        "frame_path": frame_info["path"],
                        "frame_index": frame_info["frame_index"],
                        "azimuth_deg": frame_info["azimuth_deg"],
                        "yaw_rad": frame_info["yaw_rad"],
                        "error": str(e),
                    }
                )

        ply_orbit_tags = self._aggregate_tags(results)
        summary = {
            "ply_path": ply_path,
            "num_views": self.num_views,
            "orbit_axis": "world_Y",
            "total_frames_analyzed": len(results),
            "analysis_time": datetime.now().isoformat(),
            "ply_orbit_tags": ply_orbit_tags,
            "frame_analysis": results,
            "summary": self._generate_summary(results),
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n✅ 点云环绕打标完成！结果保存到: {output_file}")
        print(f"🎯 聚合标签: {', '.join(ply_orbit_tags[:10])}...")
        return summary

    def _aggregate_tags(self, frame_results: list) -> list:
        tag_freq = {}
        for result in frame_results:
            if "tags" in result:
                for tag in result["tags"]:
                    tag_freq[tag] = tag_freq.get(tag, 0) + 1
        sorted_tags = sorted(tag_freq.items(), key=lambda x: x[1], reverse=True)
        return [tag for tag, _freq in sorted_tags[:20]]

    def _generate_summary(self, frame_results: list) -> str:
        descriptions = []
        for result in frame_results:
            if "description" in result:
                descriptions.append(result["description"])
        if descriptions:
            s = "点云（环绕视角渲染）内容摘录：\n"
            for i, desc in enumerate(descriptions[:5]):
                s += f"  {i + 1}. {desc[:100]}...\n"
            return s
        return "未能生成点云摘要"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PLY 点云绕 Y 轴均匀视角渲染 + DeepSeek-VL/BLIP-2 逐图打标（参数风格对齐 video_extract_frames_tagger）"
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
        help="模型类型：deepseek-vl 或 blip2",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="设备：cuda 或 cpu；不传则自动检测",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="每图描述最大 token 数，默认 512（与 image_tagger 一致）",
    )
    parser.add_argument(
        "--ply",
        type=str,
        required=True,
        help="输入 .ply 点云路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="汇总结果 JSON；未指定时为「PLY 同级目录 / 文件名_模型名.json」",
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        default=None,
        help="渲染图目录；未指定时为「PLY 同级目录 / 文件名_模型名_orbit_frames」",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=12,
        help="绕 Y 轴均匀采样多少个方位角（默认 12）",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=120_000,
        help="渲染用最大点数，超过则随机下采样（默认 120000）",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=960,
        help="渲染图宽度（像素），默认 960",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="渲染图高度（像素），默认 720",
    )
    parser.add_argument(
        "--view_elev",
        type=float,
        default=18.0,
        help="Matplotlib view_init 仰角 elev（度），默认 18",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.ply):
        print(f"PLY 不存在或不是文件: {args.ply}")
        raise SystemExit(1)

    tagger = PlyOrbitFrameTagger(
        model_path=args.model_path,
        model_type=args.model_type,
        device=args.device,
        max_new_tokens=args.max_tokens,
        num_views=args.num_views,
        max_points=args.max_points,
        render_width=args.width,
        render_height=args.height,
        view_elev=args.view_elev,
    )

    ply_abs = os.path.abspath(os.path.normpath(args.ply))
    ply_dir = os.path.dirname(ply_abs)
    ply_stem = os.path.splitext(os.path.basename(ply_abs))[0]
    model_suffix = getattr(tagger.image_tagger, "model_name_suffix", "default")

    output_file = args.output
    if output_file is None:
        output_file = os.path.join(ply_dir, f"{ply_stem}_{model_suffix}.json")

    frames_dir = args.frames_dir
    if frames_dir is None:
        frames_dir = os.path.join(ply_dir, f"{ply_stem}_{model_suffix}_orbit_frames")

    print(f"汇总 JSON: {output_file}")
    print(f"渲染图目录: {frames_dir}")

    tagger.analyze_ply(
        ply_abs,
        output_file=output_file,
        frames_dir=frames_dir,
    )
