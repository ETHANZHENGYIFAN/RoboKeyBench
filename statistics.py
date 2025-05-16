import json
from typing import Dict, Any
import re

PRIMITIVE_TYPES = {
    "Points": ["Anchor Point", "Grasp Point", "Actuation Point", "Hinge Point"],
    "Axes": ["Primary Axis", "Functional Axis", "Approach Axis", "Rotation Axis"]
}

def analyze_results(data: list) -> Dict[str, Any]:
    # 初始化统计器
    stats = {
        "Type Identification": {"correct": 0, "total": 0},
        "Task-to-Primitive Mapping": {"correct": 0, "total": 0},
        "Task Association": {"correct": 0, "total": 0},
        "Overall": {"correct": 0, "total": 0}
    }
    
    # 预生成类型检查列表
    primitive_list = [item for sublist in PRIMITIVE_TYPES.values() for item in sublist]
    
    for item in data:
        if item["status"] != "success":
            continue
            
        # 基础检查
        gt = item["ground_truth"]
        pred = item["prediction"][1:]
        is_correct = pred in gt.split(", ") if "," in gt else pred == gt
        
        # 任务分类
        task_type = None
        base_pred = re.match(r'^(point|axis)\b', gt.lower())

        if gt in primitive_list:
            task_type = "Type Identification"
        elif base_pred and base_pred.group(1) in ['point', 'axis']:
            task_type = "Task-to-Primitive Mapping" 
        else:
            task_type = "Task Association"
            
        # 更新统计
        stats[task_type]["total"] += 1
        stats["Overall"]["total"] += 1
        if is_correct:
            stats[task_type]["correct"] += 1
            stats["Overall"]["correct"] += 1
            
    # 计算准确率
    for k in stats:
        if k == "Overall": continue
        stats[k]["accuracy"] = round(
            (stats[k]["correct"] / stats[k]["total"] * 100) if stats[k]["total"] else 0,
            2
        )
    stats["Overall"]["accuracy"] = round(
        (stats["Overall"]["correct"] / stats["Overall"]["total"] * 100) if stats["Overall"]["total"] else 0,
        2
    )
    
    return stats


with open(r".\code\results\gpt4o\predictions_test.json") as f:
    data1 = json.load(f)
with open(r".\code\results\gpt4o\predictions_val.json") as f:
    data2 = json.load(f)
    
data = data1+data2

stats = analyze_results(data)
print(json.dumps(stats, indent=2))