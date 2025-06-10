from openai import AsyncOpenAI

from data.assistant import AssistantEntity
from utils.security import decrypt
from utils.config import config
from server.response import SuccessResponse
from pydantic import BaseModel
import json
from utils.log import log
from datetime import datetime


QUESTIONS = [
    ("品牌权威性质询", "你们和其他旅行社有什么区别？为什么你们的欧洲游产品比携程少3天？", 5),
    ("签证保险疑问", "保险是否包含新冠治疗？境外医院如何理赔？", 10),
    ("目的地咨询", "北海道10月的天气适合穿什么？", 10),
    ("特殊群体关怀", "带3岁孩子参加，酒店能提供婴儿床吗？", 15),
    ("行程细节追问", "大巴车有没有USB充电口？", 5),
    ("个性化需求", "海鲜过敏，能否调整菜单？", 5),
    ("突发情况应对", "出发前一周签证没办下来怎么办？", 10),
    ("竞品对比", "携程承诺五星酒店，你们为什么只安排四星？", 15),
    ("价格敏感型提问", "比其他平台贵500元，能再优惠吗？", 20),
    ("售后问题", "行李损坏如何理赔？", 5)
]

def format_questions(questions):
    """将问题列表格式化为可读字符串"""
    return "\n".join([
        f"- 问题类型：{q_type} | 问题举例：{example} | 在整体提问中，本类问题出现概率:{weight}%" 
        for q_type, example, weight in questions
    ])

def get_current_date():
    # 获取当前日期
    current_date = datetime.now()

    # 按照年月日格式输出
    formatted_date = current_date.strftime("%Y年%m月%d日")

    return formatted_date

def get_prompt(role):
    # 动态格式化问题列表
    formatted_questions = format_questions(QUESTIONS)
    current_date = get_current_date()
    
    prompt = f"""
# 角色指令
- 你正扮演出境游客户，向众信旅行公司咨询。
- 请严格遵守：你永远是客户（assistant），绝对不扮演客服（user）。
- 你的对话内容要符合一个实际的旅行场景。
- 你需要先了解行程，然后询问价格，之后再根据问题生成规则进行提问。
- 当前日期是{current_date}，你的需求要符合当前日期，比如冬天不要问夏天的出行问题。
- 禁止不考虑客服问题，直接提新的问题。

# 客户人设配置
{role}

# 问题生成规则
按照问题类型生成问题，每个问题类型都给出了例子用于参考，并给出了该类问题整体出现的概率。
在回答问题前，需要保证该问题符合旅行场景，并且要结合聊天历史。
如果客服在历史聊天中有问题，你需要先回复再提问。
不要问已经问过且得到回答的问题。
{formatted_questions}

# 提问风格：
- 初级难度：简单直接，语气友好
- 中级难度：带有比较性提问（如竞品对比）
- 高级难度：强势追问，要求提供解决方案

# 语气要求：
- 用词选择（如挑剔型客户使用"你们连这个都做不到？"）
- 句式结构（如强势型客户多用反问句）

# 交互规范：
- 每次只提1个问题
- 问题需包含具体场景细节（如带宠物旅行需明确宠物品种）
- 能够根据客服反馈进行场景化提问
   
# 终止条件
当出现以下任一情况时结束对话：
- 客服完整解答全部提问内容
- 客服多次无法提供有效信息
- 你已经没有更多可以提问的内容

# 限制
- 你是客户，不是客服，注意不要混淆角色
- 必须模拟一个真实的客户的对话
- 必须有明确的目的地，目的地可以是国外任何城市和景点
- 必须结合旅行场景，不允许生硬的介绍自己或者旅行需求
- 必须响应客服的问题，不允许生硬的提问
- 如果客服询问了问题，你必须要回答
- 绝对不允许询问相同问题
- 禁止不考虑客服问题，直接提新的问题。

"""
    return prompt.strip()


TERMINATION_PROMPT = """
角色定义：
user：表示一个旅行客服
assistant：表示一个旅行客户

请分析当前旅行咨询对话是否需要结束。判断依据：
1. 客户已获得所有必要信息（行程/费用/住宿）
2. 已经和客户沟通结束，客户已经说结束语
3. 客服表现得非常不好，客户表示不想接着聊下去了


只需返回true/false，不要解释。
当前对话：
{recent_dialog}
"""


async def should_terminate(messages: list) -> bool:
    """简化版终止判断"""
    try:
        # 硬性条件：对话轮次>=30
        if len([m for m in messages if m["role"] == "user"]) >= 30:
            return True

        # 智能判断
        recent_dialog = messages

        client = AsyncOpenAI(
            api_key=decrypt(config['api_key']),
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        )
        response = await client.chat.completions.create(
            model="qwen-max",
            messages=[{
                "role": "system",
                "content": TERMINATION_PROMPT.format(recent_dialog=recent_dialog)
            }],
            temperature=0.1
        )

        return response.choices[0].message.content.strip().lower() == "true"
    except Exception:
        return False


async def get_text_message(task_id):
    log.info(f'get_text_message query_task: {task_id}')
    assistant_entity = AssistantEntity.query_first(task_id=task_id)
    messages = [
        {
            'role': 'system',
            'content': get_prompt(assistant_entity.role)
        }
    ]
    messages = messages + json.loads(assistant_entity.messages) if assistant_entity.messages else []

    log.info(f'get_text_message query_task: {task_id}, message: {messages}')
    
    full_response = []
    try:
        client = AsyncOpenAI(
            api_key=decrypt(config['api_key']),
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        )

        completion = await client.chat.completions.create(
            model="qwen-max",
            messages=messages,
            stream=True,
            stream_options={"include_usage": True}
        )

        # 流式传输
        async for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                log.info(f'get_text_message response text: {content}')
                full_response.append(content)
                yield content

        # 完整响应后判断终止
        if full_response:
            assistant_response = "".join(full_response)
            messages.append({"role": "assistant", "content": assistant_response})

            if await should_terminate(messages):
                yield "[END]"

    finally:
        # 不强制发送结束标记
        pass