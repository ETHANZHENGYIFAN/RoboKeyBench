import trimesh
from xml.etree import ElementTree as ET
import os
import numpy as np
import json
import copy
from trimesh.ray import ray_triangle
from trimesh.viewer import SceneViewer
from trimesh.scene import Camera
from scipy.spatial.distance import cdist
from PIL import Image

def convert_640_to_800(pixel_coords, original_size=640, target_size=800):
    """
    将 640x640 像素坐标转换为 800x800 像素坐标。

    参数:
    - x_640, y_640: 原始图像的像素坐标 (640x640)。
    - original_size: 原始图像的分辨率，默认 640。
    - target_size: 目标图像的分辨率，默认 800。

    返回:
    - (x_800, y_800): 转换后的像素坐标 (800x800)。
    """
    converted_pixel_coord = []
    labels = []
    for pixel_coord in pixel_coords:
        x_640, y_640 = pixel_coord['coordinates']
        scale = target_size / original_size  # 计算缩放比例
        converted_pixel_coord.append((x_640 * scale, y_640 * scale))
        labels.append(pixel_coord['label'])
    return converted_pixel_coord, labels

def add_axis_to_scene(scene, main_point, visual=False):
    main_point = main_point[0]
    
    # 定义X轴
    x_end_point = main_point + np.array([0.7, 0, 0])  # X轴方向
    x_line = trimesh.load_path(np.hstack([main_point, x_end_point]).reshape(-1, 2, 3), colors=[[255, 0, 0, 255]])
    scene.add_geometry(x_line)
    x_end_point_r = main_point - np.array([0.7, 0, 0])  # X轴方向
    x_line_r = trimesh.load_path(np.hstack([main_point, x_end_point_r]).reshape(-1, 2, 3), colors=[[255, 0, 0, 255]])
    scene.add_geometry(x_line_r)
    
    # 定义Y轴
    y_end_point = main_point + np.array([0, 0.7, 0])  # Y轴方向
    y_line = trimesh.load_path(np.hstack([main_point, y_end_point]).reshape(-1, 2, 3), colors=[[0, 255, 0, 255]])
    scene.add_geometry(y_line)
    y_end_point_r = main_point - np.array([0, 0.7, 0])  # Y轴方向
    y_line_r = trimesh.load_path(np.hstack([main_point, y_end_point_r]).reshape(-1, 2, 3), colors=[[0, 255, 0, 255]])
    scene.add_geometry(y_line_r)
    
    # 定义Z轴
    z_end_point = main_point + np.array([0, 0, 0.7])  # Z轴方向
    z_line = trimesh.load_path(np.hstack([main_point, z_end_point]).reshape(-1, 2, 3), colors=[[0, 0, 255, 255]])
    scene.add_geometry(z_line)
    z_end_point_r = main_point - np.array([0, 0, 0.7])  # Z轴方向
    z_line_r = trimesh.load_path(np.hstack([main_point, z_end_point_r]).reshape(-1, 2, 3), colors=[[0, 0, 255, 255]])
    scene.add_geometry(z_line_r)
    
    if visual:
        SceneViewer(scene)


def add_keypoints_to_scene(scene, keypoints_3d, color=[255, 0, 0, 255], visual=False):
    """
    将 3D 关键点添加到场景中。
    参数:
        - scene: Trimesh 场景对象
        - keypoints_3d: 关键点的 3D 坐标列表 [(x, y, z), ...]
    """
    for idx, point in enumerate(keypoints_3d):
        sphere = trimesh.creation.icosphere(radius=0.005, subdivisions=3)  # 创建一个小球体
        sphere.visual.vertex_colors = color  
        sphere.apply_translation(point)  
        scene.add_geometry(sphere, node_name=f"Keypoint_{idx}")
    
    if visual:             
        SceneViewer(scene)

def pixel_to_3d_batch(scene, pixel_coords, labels, view):
    """
    批量处理像素点，计算 3D 坐标并返回。
    参数:
        - scene: Trimesh 场景对象
        - pixel_coords: 像素坐标列表 [(x, y), ...]
    返回:
        - 3D 坐标列表 [(x, y, z), ...]
    """
    
    keypoints_3d_c = []
    keypoints_3d_f = []
    x, y, z = view
    
    K = copy.deepcopy(scene.camera.K)
    camera_transform = copy.deepcopy(scene.camera_transform)
    R_inv = np.linalg.inv(camera_transform[:3, :3])
    ray_origin = camera_transform[:3, 3]
      
    # sphere = trimesh.creation.icosphere(radius=0.005, subdivisions=3)  # 创建一个小球体
    # sphere.visual.vertex_colors = [255, 255, 255, 255]
    # sphere.apply_translation(ray_origin)  # 将球体移动到关键点位置
    # scene.add_geometry(sphere)
    
    # line = trimesh.load_path(np.hstack([ray_origin, ray_origin + np.array([0, 0, 1])]).reshape(-1, 2, 3), colors=[[255, 0, 0, 255]])
    # scene.add_geometry(line)
    # line = trimesh.load_path(np.hstack([ray_origin, ray_origin + np.array([0, 1, 0])]).reshape(-1, 2, 3), colors=[[0, 255, 0, 255]])
    # scene.add_geometry(line)
    # line = trimesh.load_path(np.hstack([ray_origin, ray_origin + np.array([1, 0, 0])]).reshape(-1, 2, 3), colors=[[0, 0, 255, 255]])
    # scene.add_geometry(line)

    
    for pixel_coord,label in zip(pixel_coords, labels):
        u, v = pixel_coord

        # 1. 像素坐标转相机坐标
        pixel_coords = np.array([
            (u - K[0, 2]) / K[0, 0],  # x 方向归一化
            (v - K[1, 2]) / K[1, 1],  # y 方向归一化
            1                          # z 恒为 1
        ])

        # 2. 相机坐标转世界坐标
        ray_direction = R_inv @ pixel_coords
        
        if z == 180:
            ray_direction[0] = -ray_direction[0]
        if z == 270:
            ray_direction = ray_direction[[1, 2, 0]]
            ray_direction[1] = ray_direction[1]
        if z == 90:
            ray_direction = ray_direction[[1, 2, 0]]
            ray_direction[1] = -ray_direction[1]
        if x == 0:
            ray_direction[2] = -ray_direction[2]
            ray_direction[1] = -ray_direction[1]
        if x == 180:
            ray_direction[2] = -ray_direction[2]
            
        ray_direction_r = -ray_direction

        # 3. 射线得到交点
        meshes = [geom for geom in scene.geometry.values()]
        combined_mesh = trimesh.util.concatenate(meshes)
        intersector = ray_triangle.RayMeshIntersector(combined_mesh)

        locations, _, _= intersector.intersects_location(
            ray_origins=[ray_origin],
            ray_directions=[ray_direction],
            ray_tolerance=1e-5,
            multiple_hits=True
        )

        if len(locations) > 0:
            # 按距离排序并取第一个交点
            distances = np.linalg.norm(locations - ray_origin, axis=1)
            sorted_indices = np.argsort(distances)
            if label == "C":
                keypoints_3d_c.append(locations[sorted_indices][0])
            if label == "F":
                keypoints_3d_f.append(locations[sorted_indices][0])
                
        else:
            locations, _, _= intersector.intersects_location(
            ray_origins=[ray_origin],
            ray_directions=[ray_direction_r],
            ray_tolerance=1e-5,
            multiple_hits=True
        )
            if len(locations) > 0:
                locations = locations
                # 按距离排序并取第一个交点
                distances = np.linalg.norm(locations - ray_origin, axis=1)
                sorted_indices = np.argsort(distances)
                if label == "C":
                    keypoints_3d_c.append(locations[sorted_indices][0])
                if label == "F":
                    keypoints_3d_f.append(locations[sorted_indices][0])
            else:
                if label == "C":
                    keypoints_3d_c.append(None)
                if label == "F":
                    keypoints_3d_f.append(None)
                    
        # ray_origin_end_point = ray_origin - ray_direction * 10  # X轴方向
        # line = trimesh.load_path(np.hstack([ray_origin, ray_origin_end_point]).reshape(-1, 2, 3), colors=[[0, 255, 0, 255]])
        # scene.add_geometry(line)
        
        # ray_origin_end_point = ray_origin + ray_direction * 10  # X轴方向
        # line = trimesh.load_path(np.hstack([ray_origin, ray_origin_end_point]).reshape(-1, 2, 3), colors=[[0, 255, 0, 255]])
        # scene.add_geometry(line)
            
        # SceneViewer(scene)

    return keypoints_3d_c, keypoints_3d_f

def merge_close_points(points, threshold=0.01, mean=False):
    """
    合并距离小于 threshold 的点，使用均值替代。
    
    参数：
        points (list 或 np.ndarray): 形状为 (N, 3) 的 3D 点列表
        threshold (float): 距离阈值，小于该值的点会合并
        
    返回：
        np.ndarray: 消歧后的关键点列表
    """
    points = np.array(points)  # 确保 points 是 NumPy 数组
    num_points = len(points)

    # 计算所有点两两之间的欧几里得距离
    distance_matrix = cdist(points, points)  # 生成 N x N 的距离矩阵
    visited = set()  # 用于存储已合并的点索引
    merged_points = []

    for i in range(num_points):
        if i in visited:
            continue  # 如果该点已经被归入某个簇，则跳过

        # 找到所有距离小于 threshold 的点
        close_indices = np.where(distance_matrix[i] < threshold)[0]

        # 计算这些点的均值
        if mean:
            merged_point = np.mean(points[close_indices], axis=0)
        else:
            merged_point = points[close_indices][0]
        merged_points.append(merged_point)

        # 标记这些点已被合并
        visited.update(close_indices)

    return np.array(merged_points)

# ----------------------------- 主程序 -----------------------------
# TODO：这里修改文件路径
# objects = ["alcohol", "candle", "fork", "kettle", "mug", "pan", "plate", "scissors", "spoon", "teapot"]
objects = ["kettle"]

for object in objects:
    object_path = rf".\code\source\{object}"
    obj_path = sorted([folder for folder in os.listdir(object_path) if os.path.isdir(os.path.join(object_path, folder))])
    
    for obj in obj_path:
        try:
            xml_file = rf".\code\source\{object}\{obj}\model.xml"
            obj_folder = rf".\code\source\{object}\{obj}\visual"
            texture_folder = rf".\code\source\{object}\{obj}"

            # 解析 XML 文件
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # 构建 mesh 名称到文件路径的映射
            main_points = []
            mesh_name_to_file = {}
            for mesh in root.findall(".//mesh"):
                mesh_name = mesh.attrib.get("name")
                mesh_file = mesh.attrib.get("file")
                if mesh_name and mesh_file:
                    mesh_name_to_file[mesh_name] = mesh_file
                    
            # 提取材质-纹理映射
            material_to_texture = {}
            for material in root.findall(".//material"):
                material_name = material.attrib["name"]
                texture_name = material.attrib.get("texture")
                if texture_name:
                    material_to_texture[material_name] = texture_name

            # 提取纹理路径
            texture_files = {}
            for texture in root.findall(".//texture"):
                texture_name = texture.attrib["name"]
                texture_file = os.path.join(texture_folder, texture.attrib["file"])
                texture_files[texture_name] = texture_file
                    
            # 定义旋转角度 (angle_x, angle_y, angle_z)
            standard_views = [
                (90, 0, 0),       # 正面 Front: 水平朝前 (0, -2, 0)
                (90, 0, 90),      # 右侧 Right: 水平右侧 (2, 0, 0)
                (90, 0, 180),     # 背面 Back: 水平朝后
                (90, 0, 270),     # 左侧 Left: 水平左侧
                (0, 0, 0),        # 顶视 Top: 从上往下看
                (180, 0, 0)       # 底视 Bottom: 从下往上看
            ]

            # 创建场景
            scene = trimesh.Scene()
            # 加载每个 geom 的 mesh
            for geom in root.findall(".//geom"):
                mesh_name = geom.attrib.get("mesh")
                if not mesh_name:
                    continue

                # 跳过 collision 文件
                if "collision" in mesh_name:
                    continue

                # 根据 mesh_name 找到对应的文件
                mesh_file = mesh_name_to_file.get(mesh_name)
                if mesh_file:
                    obj_file = os.path.join(obj_folder, os.path.basename(mesh_file))
                    if os.path.exists(obj_file):
                        mesh = trimesh.load(obj_file)
                        # 加载纹理
                        material_name = geom.attrib.get("material")
                        if material_name in material_to_texture:
                            texture_name = material_to_texture[material_name]
                            if texture_name in texture_files:
                                texture_file = texture_files[texture_name]
                                if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
                                    # 使用 Pillow 加载纹理
                                    image = Image.open(texture_file)
                                    mesh.visual = trimesh.visual.TextureVisuals(uv=mesh.visual.uv, image=image)
                        scene.add_geometry(mesh)
            
            valid_keypoints_3ds_c = []
            valid_keypoints_3ds_f = []

            # TODO：这里改变图片的角度
            for view in standard_views:
                angle_x, angle_y, angle_z = view[0], view[1], view[2]
                points_folder = rf".\code\output\viewpoint\{object}\{obj}\{obj}_x{angle_x}_y{angle_y}_z{angle_z}_results.json"

                # 读取JSON文件
                with open(points_folder, 'r') as f:
                    points = json.load(f)

                # 提取关键点并按数字顺序排序
                keypoints_data = points.get('keypoints', {})
                sorted_keys = sorted(keypoints_data.keys(), key=lambda x: int(x))  # 按数字顺序排序键
                pixel_coord = [keypoints_data[key] for key in sorted_keys]
            
                # 生成旋转矩阵
                transform = trimesh.transformations.euler_matrix(
                    np.radians(angle_x),
                    np.radians(angle_y),
                    np.radians(angle_z)
                )

                # 定义分辨率
                width, height = 800, 800
                # 定义内参矩阵
                fx, fy = 1000, 1000
                cx, cy = 400, 400
                K = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])

                camera = Camera(resolution=(width, height), focal=(fx, fy))

                # 设置相机位置
                camera_pose = np.eye(4)                 # 初始化相机位姿矩阵
                camera_pose[:3, :3] = transform[:3, :3] # 设置相机旋转方向
                camera_pose[:3, 3] = transform[:3, 2] * 2  # 根据旋转方向设置相机位置
                
                # 添加相机节点到场景
                scene.camera = camera
                scene.camera_transform = camera_pose

                # 计算主轴点
                if view == (0, 0, 0):
                    main_point_0 = pixel_coord[0]
                    main_point_0, label_0 = convert_640_to_800([pixel_coord[0]])
                    main_keypoint_0, _ = pixel_to_3d_batch(scene, main_point_0, label_0, view)
                    
                if view == (180, 0, 0):
                    main_point_1 = pixel_coord[0]
                    main_point_1, label_1 = convert_640_to_800([pixel_coord[0]])
                    main_keypoint_1, _ = pixel_to_3d_batch(scene, main_point_1, label_1, view)
                    
                pixel_coord, label = convert_640_to_800(pixel_coord)
                keypoints_3d_c, keypoints_3d_f = pixel_to_3d_batch(scene, pixel_coord, label, view)
                keypoints_3d_c = [_ for _ in keypoints_3d_c if _ is not None]
                keypoints_3d_f = [_ for _ in keypoints_3d_f if _ is not None]
                keypoints_3d_c = np.array(keypoints_3d_c) if keypoints_3d_c else np.empty((0, 3))
                keypoints_3d_f = np.array(keypoints_3d_f) if keypoints_3d_f else np.empty((0, 3))
                filtered_points = np.concatenate((keypoints_3d_c, keypoints_3d_f), axis=0)
            
                # 过滤有效的 3D 点并添加到场景
                valid_keypoints_3ds_c.extend(_ for _ in keypoints_3d_c)
                valid_keypoints_3ds_f.extend(_ for _ in keypoints_3d_f)
            
            # valid_keypoints_3ds_c = np.array(valid_keypoints_3ds_c) if valid_keypoints_3ds_c else np.empty((0, 3))
            # valid_keypoints_3ds_f = np.array(valid_keypoints_3ds_f) if valid_keypoints_3ds_f else np.empty((0, 3))    
            keypoints_points = np.concatenate((valid_keypoints_3ds_c, valid_keypoints_3ds_f), axis=0)
            main_keypoint = (np.array(main_keypoint_0) + np.array(main_keypoint_1)) / 2

            # c是主点，f是特征点
            filtered_points_c = merge_close_points(valid_keypoints_3ds_c, threshold=0.1, mean=True)
            filtered_points_f = merge_close_points(valid_keypoints_3ds_f, threshold=0.2)
            # filtered_points_c = np.array(filtered_points_c) if filtered_points_c else np.empty((0, 3))
            # filtered_points_f = np.array(filtered_points_f) if filtered_points_f else np.empty((0, 3))
            filtered_points = np.concatenate((filtered_points_c, filtered_points_f), axis=0)
            
            add_axis_to_scene(scene, main_keypoint, visual=False)
            add_keypoints_to_scene(scene, filtered_points, visual=False)
            
            # TODO:
            # 从6个角度拍摄物体，并输出
            standard_views = [
            (90, 0, 0),    # 正面
            (90, 0, 90),   # 右侧
            (90, 0, 180),  # 背面
            (90, 0, 270),  # 左侧
            (0, 0, 0),     # 顶部
            (180, 0, 0)    # 底部
            ]

            # 创建输出目录
            output_dir = r".\code\output\3d_view"
            os.makedirs(output_dir, exist_ok=True)

            # 循环每个视角渲染图像
            for angle_x, angle_y, angle_z in standard_views:
                # ---------------------- 设置相机位姿 ----------------------
                # 生成当前视角的旋转矩阵
                transform = trimesh.transformations.euler_matrix(
                    np.radians(angle_x),
                    np.radians(angle_y),
                    np.radians(angle_z)
                )
                
                # 创建新的相机对象
                camera = Camera(resolution=(800, 800), focal=(1000, 1000))
                
                # 设置相机外参
                camera_pose = np.eye(4)
                camera_pose[:3, :3] = transform[:3, :3]  # 旋转部分
                
                # 控制相机距离：将相机放置在物体前方2米位置
                distance = 2.0  # 根据物体尺寸调整
                camera_pose[:3, 3] = transform[:3, :3] @ np.array([0, 0, distance])  # 平移部分
                
                # 更新场景相机
                scene.camera = camera
                scene.camera_transform = camera_pose
                
                # ---------------------- 渲染设置 ----------------------
                # 设置渲染参数
                render_args = {
                    'resolution': (800, 800),
                    'visible':True,          # 关闭实时窗口显示以加速
                    'flags': {'cull': False}, # 禁用背面剔除
                    'background': [255,255,255,255]  # 白色背景
                }
                
                # ---------------------- 保存图像 ----------------------
                # 生成文件名
                filename = f"{obj}_x{angle_x}_y{angle_y}_z{angle_z}.png"
                output_path = os.path.join(output_dir, filename)
                
                # 渲染并保存图像
                image_bytes = scene.save_image(**render_args)
                # scene.show()
                with open(output_path, 'wb') as f:
                    f.write(image_bytes)
                
                print(f"Saved: {output_path}")

            print("All views rendered!")
        except Exception as e:
            print(f"Error: {e}")
            print(f"Object: {obj}")