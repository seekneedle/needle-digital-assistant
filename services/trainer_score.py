from openai import OpenAI
from pydantic import BaseModel
from typing import List
from utils.security import decrypt
from utils.config import config
from server.response import SuccessResponse


class TrainerScoreRequest(BaseModel):
    messages: List[object]


class TrainerScoreResponse(BaseModel):
    content: str


def get_prompt(context):
    return f'''下面有一段通话内容，请阅读通话内容，并按照提供的评分规则对通话内容进行打分。

通话内容：
{context}

请对通话内容进行评分并输出，满分为100分，最低为0分。如出现问号，句号，感叹号则视为一句话。不要加入其它判断标准并请严格按照我下面提供的标准打分。
评分规则为：若用户对推荐的产品不满意，扣5分。
如果客服说脏话，扣5分。
如果客服没有回答用户合理问题，扣5分。

请输出最后的评分，涉及到的通话录音中扣分的句子以及触及的扣分项目中的句子。'''


async def trainer_score(request: TrainerScoreRequest):
    content = get_prompt(request.messages)
    messages = [{'role': 'user', 'content': content}]
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
            response = SuccessResponse(data=TrainerScoreResponse(content=chunk.choices[0].delta.content)).model_dump_json()
            yield f"data: {response}\n\n"
