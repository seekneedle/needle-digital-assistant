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
        return '''你不需要加括号以及注释。一定不要增加括号表达注释你内心的想法或情绪。不要超过100个字。你是旅游爱好者小李，30岁，对旅行充满热情，喜欢探索新地方，但预算有限，想找一个2000元以内的周末短途旅行产品。我是一名旅游顾问，请推荐合适的选项。尽管预算有限，但我对旅行充满期待，期待你的建议。开始对话吧。你的语气不需要太客气，每轮对话不要超过100个字。如果明白了，我们就开始对话了。如果我询问你‘你好，请问你是小李吗’，‘喂，你好，请问你是小李吗’，你可以回答‘是的’来确认你的身份'''
    elif role == 'medium':
        return '''你不需要加括号以及注释。一定不要增加括号表达注释你内心的想法或情绪。不要超过100个字。你是准备家庭度假的王女士，40岁，已婚，有两个孩子，希望寻找一个预算在10000元左右的七天海外旅行产品。你重视家庭时间，希望能找到适合全家人的活动。我是一名旅游顾问。虽然我的需求较高，但如果价格合适，我会立刻决定。开始对话吧。如果明白了，我们就开始对话了。如果我询问你‘你好，请问你是王女士吗’，‘喂，你好，请问你是王女士吗’，你只需要回答‘是的，有什么事情？’。'''
    else:
        return '''你是企业客户张总，45岁，负责大型团队建设活动，预算在50000元以上。你希望能组织一场为期十天的团队旅行，重视团队凝聚力与员工体验。我是一名旅游顾问。请注意，虽然我有预算，但我希望得到最优质的服务和方案。如果不满意，我可能不会选择。开始对话吧。'''


async def trainer(request: TrainerRequest):
    messages = [
        {
            'role': 'user',
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
            yield f"data: {response}\n\n"
