import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import csv

def get_edge(img):
    # 直接转灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 简单二值化（假设背景是浅色）
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)  # 阈值200根据实际情况调整

    # 找最外层轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 取最大轮廓（因为物体是唯一且居中的）
    contour = max(contours, key=cv2.contourArea)  # 形状为[N,1,2]
    

    # 转换为坐标列表 [[x,y],...]
    edge_points = contour.squeeze()  # 移除冗余维度 → [N,2]

    # # 输出所有坐标
    # np.savetxt("edge_coordinates.txt", edge_points, fmt="%d")

    # # 快速可视化确认
    # cv2.drawContours(img, [contour], -1, (0,255,0), 2)
    # cv2.imshow('Result', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return edge_points

def find_nearest(point, candidates):
    """找到最近的候选点"""
    distances = np.linalg.norm(candidates - point, axis=1)
    min_idx = np.argmin(distances)
    return candidates[min_idx], distances[min_idx]

def find_adjust(center, current_point, edge_points):
    """找到对应角度的候选点"""
    
    adjust_deg = 180
    adjust_point = None
    
    # 计算方向向量
    vec_current = (current_point[0] - center[0], current_point[1] - center[1])
    
    for edge_point in edge_points:
        vec_adjusted = (edge_point[0] - center[0], edge_point[1] - center[1])
        # 计算向量长度
        len_current = math.hypot(vec_current[0], vec_current[1])
        len_adjusted = math.hypot(vec_adjusted[0], vec_adjusted[1])
        # 处理零向量特殊情况
        if len_current < 1e-9 or len_adjusted < 1e-9:
            angle_deg = 360
        # 计算点积
        dot_product = vec_current[0] * vec_adjusted[0] + vec_current[1] * vec_adjusted[1]
        # 计算余弦值（并限制在[-1,1]范围内）
        cos_theta = dot_product / (len_current * len_adjusted)
        cos_theta = max(min(cos_theta, 1.0), -1.0)  # 处理浮点误差
        # 计算实际角度
        angle_rad = math.acos(cos_theta)
        angle_deg = math.degrees(angle_rad)
        if angle_deg < adjust_deg:
            adjust_deg = angle_deg
            adjust_point = edge_point
            
    return adjust_point
        
        

def visualize(center, edge_points, target_points, adjusted_points):
    """可视化验证"""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 800)  # X轴范围
    ax.set_ylim(0, 800)  # Y轴范围
    plt.scatter(edge_points[:,0], edge_points[:,1], s=1, c='blue', label='Edge')
    plt.scatter(target_points[:,0], target_points[:,1], s=50, c='red', marker='x', label='Original')
    plt.scatter(np.array(adjusted_points)[:,1], np.array(adjusted_points)[:,2], 
                s=50, facecolors='none', edgecolors='lime', linewidths=2, label='Adjusted')
    plt.scatter(*center, s=100, marker='+', c='black', label='Center')
    plt.legend()
    plt.gca().invert_yaxis()  # 图像坐标系Y轴向下
    plt.show()

def save_adjusted(adjusted_points, save_file):
    with open(save_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 写入表头
        writer.writerow(["id", "x", "y"])
        
        # 写入数据行
        for point in adjusted_points:
            writer.writerow([point[0], point[1], point[2]])

    print(f"文件已保存至 {save_file}")

def adjust_edge(img_file, key_points_file, threshold=0.7):
    # 读取图片
    img = cv2.imread(img_file)
    edge_points = get_edge(img)
    
    # 读取关键点
    key_points = np.loadtxt(key_points_file, delimiter=',', skiprows=1, dtype=int)
    target_points = key_points[:, 1:]  # 跳过ID列
    ids = key_points[:, 0]             # 提取ID列

    # 计算中心
    # center = np.mean(edge_points, axis=0)
    center = (400, 400)

    # 处理每个点
    adjusted_points = []
    for point_id, (x, y) in zip(ids, target_points):
        current_point = np.array([x, y])
        
        # 寻找对应边缘点
        adjusted_point = find_adjust(center, current_point, edge_points)
        
        # 计算距离比例
        len_to_target = np.linalg.norm(current_point - center)
        len_to_edge = np.linalg.norm(adjusted_point - center)
        if len_to_edge == 0:  # 避免除以零
            ratio = 0
        else:
            ratio = len_to_target / len_to_edge
        
        if ratio >= threshold:
            adjusted_points.append([point_id, *adjusted_point])
        else:
            adjusted_points.append([point_id, x, y])
            
    visualize(center, edge_points, target_points, adjusted_points)
    return adjusted_points
    

if __name__ == "__main__":
    img_file = r".\code\source\alcohol\x90.0_y0.0_z0.0.png"
    key_points_file = r".\code\source\alcohol\x90.0_y0.0_z0.0_keypoints.csv"
    save_file = r".\code\source\alcohol\x90.0_y0.0_z0.0_keypoints_adjusted.csv"

    
    adjusted_points = adjust_edge(img_file, key_points_file)
    save_adjusted(adjusted_points, save_file)