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
    role: str
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
    evals = {
        '服务规范': '总分15分，评分结合是否包含标准开场白（如“您好，欢迎咨询众信旅游”）和结束语（如“感谢您的咨询”），是否服务态度敷衍。打分标准：0-6分：遗漏关键服务话术/用语错误；7-12分：准确使用标准话术但缺乏温度；13-15分：灵活结合标准话术与个性化服务',
        '需求洞察': '总分20分，评分结合用户需求识别、响应和引导（如“二老平时有登山习惯吗？”，或“您这次旅行预算时人均1万吗”，或"想象孩子触摸南极企鹅时的光芒，这是终身教育投资"）。打分标准：0-8分：仅回应表面需求；9-16分：封闭式问题确认基础需求；17-20分：通过客户资料深度挖掘需求',
        '方案定制': '总分18分，评分结合是否重复同类错误，是否虚假承诺，方案是否符合用户需求。打分标准：0-7分：推荐标准套餐；8-14分：组合调整服务模块；15-18分：场景重构设计专属行程',
        '信任建立': '总分18分，评分结合是否有引导信任建立话术，（如"您似乎有些犹豫，需要我再说明吗？"或者"您重视深度体验，这正是我们“拒绝打卡式旅行”的理念"，或者"南极之旅的真实案例+价值观植入"）。打分标准：0-7分：基础情绪识别；8-14分：结合需求说明品牌优势；15-18分：故事化表达引发共鸣',
        '异议转化': '总分15分，要求检测是否使用不礼貌用语（如“你不懂”、“你这人怎么这么啰嗦”、"这个价格已经很便宜了"之类的）。打分标准：0-6分：直接反驳客户；7-12分：提供附加补偿；13-15分：心理账户理论转化异议',
        '成交引导': '总分14分，评分结合是否积极引导顾客下单，如询问顾客是否要立即预订，或结尾时要求加顾客微信或电话号码以便后续联系，或本周签约立减2000元，或明确跟顾客说次日或稍晚点再电话联系。打分标准：0-5分：被动等待成交；6-11分：限时促销策略；12-14分：预演产品使用场景'
    }

    prompt = f'''
        有一段对话内容，其中 user 是旅游行业的客服人员，assistant 是前来咨询旅游信息的顾客。
        请阅读对话内容，并按照提供的评分维度和规则，对客服人员（user）的服务水平打分。

        通话内容：
        {context}

        # 顾客人设
        [基础信息]
        {role}

        # 评分维度和规则
        {evals}

        # 返回结果
        返回 json 对象，其中有且只有以下字段：
        scores，类型为 array[dict]，对评分规则中指定的每个维度分别打分。
          其中，每个评分维度对应 array 中的一个元素，该元素类型是 dict，有四个字段：eval（str 类型，对应评分维度），total_score（整数类型，该维度总分），score（整数类型，打分）和 reason（str 类型，这样打分的理由）。
        summary，类型为 str，对该客服人员（user）的表现做整体评价。请直接使用“你”称呼客服人员，打分结果是给客服人员看的。
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

    response = llm.qwen_call(messages, 'json_object', '', 'score', 'qwen-plus')
    log.info(f'__after llm {type(response)} {response}')
    parsed_data = json.loads(response)
    log.info(f'_json: {type(parsed_data)}, {parsed_data}')
    # Save messages to file after complete response is generated
    save_messages_to_file(request.messages, response)
    log.info(f'__before return')
    return TrainerScoreResponse(**parsed_data)
