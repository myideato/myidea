import open3d as o3d
if __name__ == "__main__":
    # 读取 .ply 点云文件 [citation:1][citation:5]
    pcd = o3d.io.read_point_cloud("ply-tag-file/xyzrgb_dragon.ply")

    # 打印信息并可视化 [citation:9]
    print(pcd)
    o3d.visualization.draw_geometries([pcd], window_name="Open3D Viewer")
