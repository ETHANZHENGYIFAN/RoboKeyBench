import json

file_path = r'.\code\results\open_source\Qwen2.5-VL-7B_results_20250514_093326.json'


with open(file_path) as f:
    data = json.load(f)

cnt_task = [0, 0, 0]
true_task = [0, 0, 0]
flag_task = 0

cnt_axis = [0, 0]
true_axis = [0, 0]
flag_axis = 0

for item in data:
    if item['ground_truth'].endswith("Point"):
        flag_axis = 0
    elif item['ground_truth'].startswith("point"):
        flag_axis = 0
    elif item['question'].startswith("What type of primitive is point"):
        flag_axis = 0   
    elif item['ground_truth'].endswith("Axis"):
        flag_axis = 1
    elif item['ground_truth'].startswith("axis"):
        flag_axis = 1
    elif item['question'].startswith("What type of primitive is axis"):
        flag_axis = 1   
    
    cnt_axis[flag_axis] += 1
    if item['correct']:
        true_axis[flag_axis] += 1
        
    if item['question'].startswith("What type of primitive is"):
        flag_task = 0
    elif item['question'].startswith("Which task is associated with"):
        flag_task = 1
    elif item['question'].startswith("Which primitive is related to"):
        flag_task = 2
    cnt_task[flag_task] += 1
    if item['correct']:
        true_task[flag_task] += 1




print(cnt_axis)
print(true_axis)
print([x / y for x, y in zip(true_axis, cnt_axis)])

print(cnt_task)
print(true_task)
print([x / y for x, y in zip(true_task, cnt_task)])
    