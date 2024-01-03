from tqdm import tqdm
from human_eval.data import write_jsonl
from implementation import *

problems = read_problems()
model = 'chatglm2-6b'

num_samples_per_task = 1

samples = []

keys = list(problems.keys())

total_iterations = num_samples_per_task * len(keys)*3

with tqdm(total=total_iterations, desc='Generating samples') as pbar:
    for _ in range(num_samples_per_task):
        for task_id in keys:
            samples.append(dict(task_id=task_id, completion=self_planning(model, problems[task_id]["prompt"])))
            pbar.update(2)

    write_jsonl(f"{model}-selfplanning.jsonl", samples)

    samples = []

    for _ in range(num_samples_per_task):
        for task_id in keys:
            samples.append(dict(task_id=task_id, completion=completion(model, problems[task_id]["prompt"])))
            pbar.update(1)

    write_jsonl(f"{model}-direct.jsonl", samples)
