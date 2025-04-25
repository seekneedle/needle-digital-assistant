import json
from datetime import datetime
import os

from openai import OpenAI
from pydantic import BaseModel
from typing import List

from server.response import SuccessResponse
from utils.config import config
from utils.log import log
from utils.security import decrypt
from utils import llm

class TrainerScoreRequest(BaseModel):
    role: dict
    messages: List[object]
    # evals: List[dict]

class TrainerScoreResponse(BaseModel):
    scores: list[dict]
    summary: str
    tags: list[str]


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


def to_prompt(role, context):
    level = role['level']
    # 提取人设维度特征（假设role字典包含这些字段）
    personality = role.get('personality', '友好型')
    personality_example = role.get('personality_example', '"您详细说说这个行程吧，我慢慢听"')
    demand = role.get('demand', '基础需求型')
    demand_example = role.get('demand_example', '"这个行程价格是多少？"')
    background = role.get('background', '普通上班族')
    background_example = role.get('background_example', '"我是刚毕业的大学生"')
    region = role.get('region', '本地化需求')
    region_example = role.get('region_example', '"行程中有没有故宫周边景点？"')

    evals = {
        '使用标准话术': '是否包含开场白（如“您好，欢迎咨询众信旅游”）和结束语（如“感谢您的咨询”）',
        '言论适当': '检测是否使用不礼貌用语（如“你不懂”）或泄露公司机密。',
        '问题回答合理': '回答是否符合场景需求。',
        '成交转化能力': '是否成功引导客户下单（如“您是否需要立即预订？”）。'
    }

    prompt = f'''
        有一段对话内容，其中 user 是旅游行业的客服人员，assistant 是前来咨询旅游信息的顾客。
        请阅读对话内容，并按照提供的评分维度和规则，对客服人员的服务水平打分。

        通话内容：
        {context}

    # 顾客人设
    [基础信息]
    难度级别：{level}
    性格特征：{personality}（示例：{personality_example}）
    核心需求：{demand}（示例：{demand_example}）
    身份背景：{background}（示例：{background_example}）
    地域特征：{region}（示例：{region_example}）

    # 评分维度和规则
    {evals}
    
    # 返回结果
    返回 json 对象，其中有且只有以下字段：
    scores，类型为 array[dict]，对评分规则中指定的每个维度分别打分，范围为 0 到 100。
      其中，每个评分维度对应 array 中的一个元素，该元素类型是 dict，有三个字段：eval（str 类型，对应评分维度），score（整数类型，打分）和 reason（str 类型，这样打分的理由）。
    summary，类型为 str，对该客服人员（user）的表现做整体评价。
    tags，类型为 array[str]，给该客服人员（user）打若干标签，如 ["熟悉业务”, "善于沟通"] 等。
'''
    return prompt
    # return f'''
    #     下面有一段通话内容，请阅读通话内容，并按照提供的评分规则对客服通话内容进行打分。
    #     其中user是客服，assistant是客户。
    #
    #     通话内容：
    #     {context}
    #
    #     请对通话内容进行评分并输出，满分为100分，最低为0分。如出现问号，句号，感叹号则视为一句话。不要加入其它判断标准并请严格按照我下面提供的标准打分。
    #     评分规则为：若用户对推荐的产品不满意，扣5分。
    #     如果客服说脏话，扣5分。
    #     如果客服没有回答用户合理问题，扣5分。
    #     如果没有使用礼貌用语回复，扣5分。
    #     如果抱怨客人，扣5分。
    #     如果沟通过程中不耐烦，回答内容过于简单笼统，扣5分。
    #     如果回答内容中包含贬义词，扣5分。
    #     如果介绍旅游产品时过分夸张，扣5分。
    #
    #     请确保最后的得分为100分减去扣减的总分数，不要出现计算错误。
    #
    #     请输出最后的评分，涉及到的通话录音中扣分的句子以及触及的扣分项目中的句子。
    # '''


async def trainer_score(request: TrainerScoreRequest):
    log.info(f'trainer_score() called.')
    content = to_prompt(request.role, request.messages)
    messages = [{'role': 'user', 'content': content}]
    log.info(f'score(): to call qwen: __{messages}__')

    response = llm.qwen_call(messages, 'json_object', 'mock_task_id', 'score', 'qwen-turbo')
    log.info(f'__after llm {type(response)} {response}')
    parsed_data = json.loads(response)
    log.info(f'_json: {type(parsed_data)}, {parsed_data}')
    # Save messages to file after complete response is generated
    save_messages_to_file(request.messages, response)
    log.info(f'__before return')
    return TrainerScoreResponse(**parsed_data)
