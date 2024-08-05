import json
import os
from tqdm import tqdm
import re

# def split_json_objects(line):
#     # 此正则表达式尝试找到正确分割多个 JSON 对象的位置
#     objects = re.split(r'}\s*,\s*{', line)
#     if len(objects) > 1:
#         # 如果发现多个对象，重新构建它们为有效的 JSON 字符串
#         return [objects[0] + '}' if i == 0 else '}' + obj + ('}' if i == len(objects)-1 else '') for i, obj in enumerate(objects)]
#     return [line]

def clean_line(line):
    # 移除所有的空字符
    return line.replace('\x00', '')

def check_action_type(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # new_file_path = file_path.replace('.jsonl', '_out.jsonl')
    line_number = 0
    with open(file_path, 'r') as file:
        for line in tqdm(file, desc="Checking lines"):
            line_number += 1
            data = json.loads(line)
            # new_file.write(line)
            action_type = data.get("action_type")
            syms = data.get("syms")
            
            if action_type is None or syms is None or len(action_type) != len(syms):
                if action_type is None:
                    print("error1")

                elif syms is None:
                    print("error2")
                else:
                    print("error3")
                    print(len(action_type))
                    print(len(syms))
                print(f"Line {line_number} does not meet criteria.")

    print(f"Checked {line_number} entries.")
    return

input_file_path = 'data/amazon/train_processed.jsonl'
check_action_type(input_file_path)
