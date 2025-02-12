from openai import OpenAI
from pydantic import BaseModel
from typing import List
from utils.security import decrypt
from utils.config import config
from server.response import SuccessResponse
import json
from datetime import datetime
import os


class TrainerScoreRequest(BaseModel):
    messages: List[object]


class TrainerScoreResponse(BaseModel):
    content: str


def get_prompt(context):
    return f'''下面有一段通话内容，请阅读通话内容，并按照提供的评分规则对客服通话内容进行打分。
其中user是客服，assistant是客户。

通话内容：
{context}

请对通话内容进行评分并输出，满分为100分，最低为0分。如出现问号，句号，感叹号则视为一句话。不要加入其它判断标准并请严格按照我下面提供的标准打分。
评分规则为：若用户对推荐的产品不满意，扣5分。
如果客服说脏话，扣5分。
如果客服没有回答用户合理问题，扣5分。
如果没有使用礼貌用语回复，扣5分。
如果抱怨客人，扣5分。
如果沟通过程中不耐烦，回答内容过于简单笼统，扣5分。
如果回答内容中包含贬义词，扣5分。
如果介绍旅游产品时过分夸张，扣5分。

请输出最后的评分，涉及到的通话录音中扣分的句子以及触及的扣分项目中的句子。'''


def save_messages_to_file(messages: List[object], response: str):
    """
    Save request messages and AI response to a JSON file with datetime-based filename.
    
    Args:
        messages (List[object]): List of message objects to save
        response (str): AI response to save
    """
    # Ensure a logs directory exists
    logs_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'trainer_score')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Generate filename with current datetime
    filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f.json")
    filepath = os.path.join(logs_dir, filename)
    
    # Prepare data to save
    data_to_save = {
        "timestamp": datetime.now().isoformat(),
        "request_messages": messages,
        "ai_response": response
    }
    
    # Write to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=2)


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
    response_content = ""
    for chunk in completion:
        if len(chunk.choices) > 0:
            delta_content = chunk.choices[0].delta.content or ""
            response_content += delta_content
            response = SuccessResponse(data=TrainerScoreResponse(content=response_content)).model_dump_json()
            yield f"data: {response}\n\n"
    
    # Save messages to file after complete response is generated
    save_messages_to_file(request.messages, response_content)
