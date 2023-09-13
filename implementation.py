# -*- coding = utf-8 -*-
# @Time : 2023/8/31 23:36
# @Author : Tan Qiyuan
# @File : implementation
import openai
from transformers import AutoTokenizer, AutoModel

from human_eval.data import read_problems
tokenizer = AutoTokenizer.from_pretrained("THUDM/codegeex2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/codegeex2-6b", trust_remote_code=True).half().cuda()

# openai.api_key = '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'
openai.api_base = "http://localhost:8000/v1"

problems = read_problems()
keys = list(problems.keys())[:10]

ONE_SHOT_PROMPT = problems[keys[0]]['prompt'] + problems[keys[0]]['canonical_solution']

FIVE_SHOT_PROMPT = ''
for key in keys[:5]:
    FIVE_SHOT_PROMPT += problems[key]['prompt'] + problems[key]['canonical_solution'] + '\n\n'


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
        prompt=prompt,
        temperature=0.2,
        top_p=0.95,
        max_tokens=300
    )
    # print(f'{prompt+chat.choices[0].text}')
    return crop_string(chat.choices[0].text)


def five_shot_prompted(prompt, model):
    chat = openai.Completion.create(
        model=model,
        prompt=[FIVE_SHOT_PROMPT, prompt],
        temperature=0,
        top_p=1,
        max_tokens=2000
    )
    # print(f'{prompt+chat.choices[0].text}')
    return chat.choices[0].text


def one_shot_prompted(prompt, model):
    chat = openai.Completion.create(
        model=model,
        prompt=ONE_SHOT_PROMPT + prompt,
        temperature=0,
        top_p=1,
        max_tokens=400
    )
    # print(chat)
    # print(f'{prompt+chat.choices[0].text}')
    return crop_string(chat.choices[0].text)


def crop_string(input_string):
    index1 = input_string.find('\ndef')
    index2 = input_string.find('\nif')
    if index1 < index2 and index1 > -1:
        return input_string[:index1]
    return input_string[:index2]


def codegeex(prompt, model_name):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs, max_length=600, top_p=0.95, temperature=0.2)
    return tokenizer.decode(outputs[0])


if __name__ == '__main__':
    prompt = problems[keys[0]]['prompt']
    print(prompt + codegeex(prompt, 'codegeex2-6b'))
