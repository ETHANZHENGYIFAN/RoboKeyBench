import trimesh
from xml.etree import ElementTree as ET
import os
import numpy as np
from trimesh.scene import Camera
from PIL import Image

def parse_mujoco_xml(xml_file, obj_prefix, category_folder, obj_folder, texture_folder):
    # 构建输出文件夹路径
    output_folder = os.path.join(category_folder, obj_prefix)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        return

    # 解析 XML 文件
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 构建 mesh 名称到文件路径的映射
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
            # print(f"Loading OBJ file: {obj_file}")
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
                
                # 添加到场景
                scene.add_geometry(mesh)
    
    # 定义旋转角度 (angle_x, angle_y, angle_z)
    standard_views = [
        (90, 0, 0),       # 正面 Front: 水平朝前
        (90, 0, 90),      # 右侧 Right: 水平右侧
        (90, 0, 180),     # 背面 Back: 水平朝后
        (90, 0, 270),     # 左侧 Left: 水平左侧
        (135, 0, 0),      # 斜上 Isometric 1: 从上向前斜看
        (135, 0, 120),    # 斜上 Isometric 2: 从上向右斜看
        (135, 0, 240),    # 斜上 Isometric 3: 从上向左斜看
        (45, 0, 0),       # 斜下 Isometric 4: 从下向前斜看
        (45, 0, 120),     # 斜下 Isometric 5: 从下向右斜看
        (45, 0, 240),     # 斜下 Isometric 6: 从下向左斜看
        (0, 0, 0),        # 顶视 Top: 从上往下看
        (180, 0, 0)       # 底视 Bottom: 从下往上看
    ]
    

    # 遍历视角，应用变换并保存图片
    for angle_x, angle_y, angle_z in standard_views:
        try:
            # 生成旋转矩阵
            transform = trimesh.transformations.euler_matrix(
                np.radians(angle_x),
                np.radians(angle_y),
                np.radians(angle_z)
            )
            
            # 定义分辨率
            width, height = 800, 800
            # 定义内参矩阵
            # fx, fy = 692.82, 692.82
            fx, fy = 1000, 1000
            cx, cy = 400, 400
            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])

            camera = Camera(resolution=(width, height), focal=(fx, fy))
            camera.K = K  # 手动覆盖内参矩阵

            # 设置相机位置
            camera_pose = np.eye(4)                 # 初始化相机位姿矩阵
            camera_pose[:3, :3] = transform[:3, :3] # 设置相机旋转方向
            camera_pose[:3, 3] = transform[:3, 2] * 2  # 根据旋转方向设置相机位置

            # 添加相机节点到场景
            scene.camera = camera
            scene.camera_transform = camera_pose

            # 渲染并保存图片
            color_image = scene.save_image(resolution=(800, 800), visible=True)

            # 生成文件名
            image_filename = f"{obj_prefix}_x{angle_x}_y{angle_y}_z{angle_z}.png"
            image_path = os.path.join(output_folder, image_filename)

            # 保存图片
            with open(image_path, 'wb') as f:
                f.write(color_image)
            print(f"已保存视角图片: {image_path}")
        except Exception as e:
            print(f"渲染视角失败: {e}")

def process_all_obj_files(input_folder, output_base_folder):
    for dirpath, dirnames, filenames in os.walk(input_folder):
        for filename in filenames:
            if filename == "model.xml":
                xml_file = os.path.join(dirpath, filename)
                obj_prefix = os.path.basename(dirpath)

                category = obj_prefix.rsplit("_", 1)[0]
                category_folder = os.path.join(output_base_folder, category)
                if not os.path.exists(category_folder):
                    os.makedirs(category_folder)
                    print(f"Created: {category_folder}")

                obj_folder = os.path.join(dirpath, "visual")
                texture_folder = dirpath
                parse_mujoco_xml(xml_file, obj_prefix, category_folder, obj_folder, texture_folder)

# 使用示例
input_folder = r".\code\source\kettle"
output_base_folder = r".\code\output\render\kettle"

# 遍历并处理所有 .obj 文件
process_all_obj_files(input_folder, output_base_folder)
