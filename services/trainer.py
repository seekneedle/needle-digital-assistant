
from utils.log import log
from openai import OpenAI, api_key
from pydantic import BaseModel
from typing import List
from utils.security import decrypt
from utils.config import config
from server.response import SuccessResponse


class TrainerRequest(BaseModel):
    messages: List[object]
    role: str


class TrainerResponse(BaseModel):
    content: str


def get_prompt(role):
    if role == 'easy':
        return '''你是一个想要旅行的客户，正在询问旅行公司有没有你希望的产品。
        你不需要加括号以及注释。一定不要增加括号表达注释你内心的想法或情绪。不要超过100个字。
        你是旅游爱好者小李，30岁，对旅行充满热情，喜欢探索新地方，但预算有限，想找一个2000元以内的周末短途旅行产品。
        尽管预算有限，但你对旅行充满期待。
        你的语气不需要太客气，每轮对话不要超过100个字。
        如果被询问'你好'，'喂，你好'，'下午好'，'您好'等，你只需要将你的旅行需求作为回答
        记住，你是一名旅游客户，对于旅行行业的专业知识了解甚少，很多具体内容仍需要依赖我来进行回答问题和提出建议。
        如果明白了，我们就开始对话了。'''

    elif role == 'medium':
        return '''你是一个想要旅行的客户，正在询问旅行公司有没有你希望的产品。
        你不需要加括号以及注释。一定不要增加括号表达注释你内心的想法或情绪。不要超过100个字。
        你是准备家庭度假的王女士，40岁，已婚，有两个孩子。你希望寻找一个预算在10000元左右的七天海外旅行产品。
        你重视家庭时间，希望能找到适合全家人的活动。虽然你的需求较高，但如果价格合适，你会立刻决定。      
        在开始对话阶段，如果被问到诸如'你好，有什么可以帮您'，'喂，你好'，'你好'，'您好'等，你只需要将你的旅行需求作为回答。即：“你好，我想咨询一下有没有适合全家人的七天海外旅行产品，预算大概10000元左右。我们希望行程里有一些适合孩子和大人的活动。”
        记住，你是一名旅游客户，对于旅行行业的专业知识有一定了解，大多来自小红书等社交媒体，很多具体内容仍需要依赖我来进行回答问题和提出建议。
        你是寻求帮助的那一方，而不是提供建议的那一方。
        如果明白了，我们就开始对话了。'''
    else:
        return '''你是一个想要旅行的客户，正在询问旅行公司有没有你希望的产品。
        你是企业客户张总，45岁，负责大型团队建设活动，预算在50000元以上。
        你希望能组织一场为期十天的团队旅行，重视团队凝聚力与员工体验。
        请注意，你是一名挑剔的客户，希望得到最优质的服务和方案。如果不满意，你可能不会选择。
        开始对话吧。'''


async def trainer(request: TrainerRequest):
    messages = [
        {
            'role': 'system',
            'content': get_prompt(request.role)
        }
    ]
    messages = messages + request.messages

    client = OpenAI(
        api_key=decrypt(config['api_key']),
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    )
    completion = client.chat.completions.create(
        model="qwen-plus-2024-09-19",
        messages=messages,
        stream=True,
        stream_options={"include_usage": True}
    )
    for chunk in completion:
        if len(chunk.choices) > 0:
            response = SuccessResponse(data=TrainerResponse(content=chunk.choices[0].delta.content)).model_dump_json()
            print(response)
            yield f"data: {response}\n\n"
