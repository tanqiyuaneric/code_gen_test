# -*- coding = utf-8 -*-
# @Time : 2023/8/31 23:36
# @Author : Tan Qiyuan
# @File : implementation
import openai
from human_eval.data import read_problems

openai.api_key = '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'
openai.api_base = "http://localhost:8000/v1"

problems = read_problems()

# ONE_SHOT_PROMPT =
def generate_one_completion(prompt):
    return planning(prompt)


def planning(prompt):
    chat = openai.Completion.create(
        model='chatglm2',
        prompt=prompt
    )
    print(f'planning: {chat.choices[0].text}')
    return chat.choices[0].text


def direct_generation(prompt, model):
    chat = openai.Completion.create(
        model=model,
        prompt=prompt
    )
    # print(f'{prompt+chat.choices[0].text}')
    return chat.choices[0].text


if __name__ == '__main__':
    print(planning('hello'))
