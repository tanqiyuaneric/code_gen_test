# -*- coding = utf-8 -*-
# @Time : 2023/8/31 23:36
# @Author : Tan Qiyuan
# @File : implementation
import openai
import zhipuai
from transformers import AutoTokenizer, AutoModel
from human_eval.data import read_problems

openai.api_key = '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'
zhipuai.api_key = '2c700bf3ba6419b8ea37f0602baf527c.7vBfCGkpob2Y8Qzr'

problems = read_problems()
keys = list(problems.keys())[:10]

ONE_SHOT_PROMPT = problems[keys[0]]['prompt'] + problems[keys[0]]['canonical_solution']

FIVE_SHOT_PROMPT = ''
for key in keys[:5]:
    FIVE_SHOT_PROMPT += problems[key]['prompt'] + problems[key]['canonical_solution'] + '\n\n'


with open('planning_prompt.txt', 'r') as file:
    # 2. 读取文件内容
    PLANNING_PROMPT = file.read()


def planning(model_name, prompt):
    print(prompt := PLANNING_PROMPT+prompt[:-4])
    return completion(model_name, prompt)


def crop_string(input_string):
    index1 = input_string.find('\ndef')
    index2 = input_string.find('\nif')
    if index2 > index1 > -1:
        return input_string[:index1]
    return input_string[:index2]


def completion(model_name, prompt, max_length=600, top_p=0.9, temperature=0.2):
    if 'codegeex' in model_name:
        return codegeex(prompt, max_length, top_p, temperature)
    elif 'pro' in model_name:
        response = zhipuai.model_api.invoke(
            model=model_name,
            prompt=prompt,
            top_p=top_p,
            temperature=temperature,
            max_length=max_length
        )
        return response['data']['choices'][0]['content']
    elif 'gpt' in model_name:
        openai.api_base = 'https://api.openai.com/v1'
        chat = openai.Completion.create(
            model=model_name,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_length
        )
        return chat.choices[0].text
    else:
        openai.api_base = "http://localhost:8000/v1"
        chat = openai.Completion.create(
            model=model_name,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_length
        )
        return chat.choices[0].text


def codegeex(prompt, max_length, top_p, temperature):
    tokenizer = AutoTokenizer.from_pretrained("THUDM/codegeex2-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/codegeex2-6b", trust_remote_code=True).half().cuda()
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs, max_length=max_length, top_p=top_p, temperature=temperature)
    return tokenizer.decode(outputs[0])


if __name__ == '__main__':
    result = planning('chatglm_pro', problems[keys[0]]['prompt'])
    print(f'response: {result}')
