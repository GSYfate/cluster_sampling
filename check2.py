import json
import os
from tqdm import tqdm

def clean_line(line):
    # 移除所有的空字符
    return line.replace('\x00', '')

def process_file_and_adjust_ids(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 创建新文件的路径
    new_file_path = file_path.replace('_out.jsonl', '_processed.jsonl')
    line_number = 0
    new_author_id = 1  # 新 author_id 从 1 开始

    with open(file_path, 'r') as file, open(new_file_path, 'w') as new_file:
        for line in tqdm(file, desc="Processing lines"):
            try:
                # 清理行并尝试解析 JSON
                line = clean_line(line)
                data = json.loads(line)

                # 检查必需字段是否存在且长度相同
                action_type = data.get("action_type")
                syms = data.get("syms")

                if action_type is None or syms is None or len(action_type) != len(syms):
                    continue  # 如果检查失败，则跳过这行

                # 更新 author_id 并写入新文件
                data['author_id'] = new_author_id
                new_author_id += 1
                new_file.write(json.dumps(data) + '\n')
                
            except json.JSONDecodeError:
                continue  # 如果 JSON 解析失败，跳过这行

    print(f"Processed file and wrote new IDs up to {new_author_id - 1}")
    return new_file_path

input_file_path = '/home/sg2323/project/authorship/data/amazon/train_filtered_out.jsonl'
new_file_path = process_file_and_adjust_ids(input_file_path)
print(f"New file created at: {new_file_path}")
