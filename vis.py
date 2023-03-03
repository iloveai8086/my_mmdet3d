import open3d as o3d
import numpy as np


def main():
    raw_point = np.fromfile('/media/ros/A666B94D66B91F4D/ros/luis/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin')  # 读取1.npy数据  N*[x,y,z]

    # 创建窗口对象
    vis = o3d.visualization.Visualizer()
    # 设置窗口标题
    vis.create_window(window_name="kitti")
    # 设置点云大小
    vis.get_render_option().point_size = 1
    # 设置颜色背景为黑色
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])

    # 创建点云对象
    pcd = o3d.open3d.geometry.PointCloud()
    # 将点云数据转换为Open3d可以直接使用的数据类型
    pcd.points = o3d.open3d.utility.Vector3dVector(raw_point)
    # 设置点的颜色为白色
    pcd.paint_uniform_color([1, 1, 1])
    # 将点云加入到窗口中
    vis.add_geometry(pcd)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
