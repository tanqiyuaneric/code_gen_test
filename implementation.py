# -*- coding = utf-8 -*-
# @Time : 2023/8/31 23:36
# @Author : Tan Qiyuan
# @File : implementation
import openai

openai.api_key = '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'


def generate_one_completion(prompt):
    print(prompt)


def planning(prompt):
    openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'system', 'content': ''}]
    )
