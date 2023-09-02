from tqdm import tqdm  # 导入tqdm库

from human_eval.data import write_jsonl, read_problems
from implementation import *

problems = read_problems()
model = 'chatglm2-6b'

num_samples_per_task = 3

samples = []

keys = list(problems.keys())

# 计算总迭代次数
total_iterations = num_samples_per_task * len(keys)

# 使用tqdm创建一个进度条
with tqdm(total=total_iterations, desc='Generating samples') as pbar:
    for _ in range(num_samples_per_task):
        for task_id in keys:
            samples.append(dict(task_id=task_id, completion=direct_generation(problems[task_id]["prompt"], model)))
            pbar.update(1)  # 更新进度条

write_jsonl("samples.jsonl", samples)
