from tqdm import tqdm  # 导入tqdm库

from human_eval.data import write_jsonl, read_problems
from implementation import *

problems = read_problems()
model = 'vicuna-7b-v1.5'

num_samples_per_task = 1

samples = []

keys = list(problems.keys())# [10:]

# 计算总迭代次数
total_iterations = num_samples_per_task * len(keys)

# 使用tqdm创建一个进度条
with tqdm(total=total_iterations, desc='Generating samples') as pbar:
    for _ in range(num_samples_per_task):
        for task_id in keys:
            samples.append(dict(task_id=task_id, completion=self_planning(model, problems[task_id]["prompt"])))
            pbar.update(1)  # 更新进度条

write_jsonl(f"{model}-selfplanning.jsonl", samples)
